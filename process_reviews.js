import sqlite3 from "sqlite3";
import { open } from "sqlite";
import { QdrantClient } from "@qdrant/js-client-rest";
import { pipeline } from "@xenova/transformers";
import { Daytona } from "@daytonaio/sdk";
import { v4 as uuidv4 } from "uuid";
import dotenv from "dotenv";

dotenv.config();

// Initialize Qdrant client
const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL || "http://localhost:6333",
  apiKey: process.env.QDRANT_API_KEY,
});

const COLLECTION_NAME = "pitchfork_reviews_daytona";
const NUM_WORKSPACES = 5;

async function setupQdrant() {
  // Create collection if it doesn't exist
  const collections = await qdrant.getCollections();
  const exists = collections.collections.some(
    (c) => c.name === COLLECTION_NAME
  );

  if (!exists) {
    await qdrant.createCollection(COLLECTION_NAME, {
      vectors: {
        size: 384, // MiniLM-L6-v2 embedding size
        distance: "Cosine",
      },
    });
  }
}

async function processReviewChunk(reviews, workspace) {
  // Install dependencies in the workspace
  console.log("Installing dependencies in workspace...");

  // Create package.json
  const packageJson = {
    name: "review-embeddings",
    version: "1.0.0",
    type: "module",
    dependencies: {
      "@xenova/transformers": "latest",
    },
  };

  const packageFile = new File(
    [JSON.stringify(packageJson, null, 2)],
    "package.json",
    { type: "text/plain" }
  );
  await workspace.fs.uploadFile("/home/daytona/package.json", packageFile);

  await workspace.process.executeCommand("npm install");

  // Create the embedding script
  const scriptCode = `
    import { pipeline } from '@xenova/transformers';
    import fs from 'fs/promises';
    
    async function main() {
      try {
        console.error('Reading input file...');
        const text = await fs.readFile('input.txt', 'utf8');
        console.error('Input text length:', text.length);
        
        console.error('Initializing embedder...');
        const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        console.error('Embedder initialized');
        
        console.error('Processing text...');
        const output = await embedder(text.trim(), {
          pooling: 'mean',
          normalize: true
        });
        console.error('Text processed');
        
        const embedding = Array.from(output.data);
        console.error('Embedding generated, length:', embedding.length);
        
        await fs.writeFile('output.json', JSON.stringify(embedding));
        console.error('Embedding saved to output.json');
        process.exit(0);
      } catch (error) {
        console.error('Error:', error);
        process.exit(1);
      }
    }
    
    main();
  `;

  // Save the script to a file
  const scriptFile = new File([scriptCode], "embed.js", { type: "text/plain" });
  await workspace.fs.uploadFile("/home/daytona/embed.js", scriptFile);

  // Process each review in the chunk
  for (const review of reviews) {
    // Skip if review is missing required fields
    if (!review || !review.review_url || !review.body) {
      console.log("Skipping invalid review:", review);
      continue;
    }

    // Only use the review body for embedding
    const textToEmbed = review.body.trim();

    if (!textToEmbed) {
      console.log(`Skipping review ${review.review_url} - no text content`);
      continue;
    }

    try {
      // Write the text to a file
      const inputFile = new File([textToEmbed], "input.txt", {
        type: "text/plain",
      });
      await workspace.fs.uploadFile("/home/daytona/input.txt", inputFile);

      // Run the embedding script
      const result = await workspace.process.executeCommand("node embed.js");

      if (result.exitCode !== 0) {
        console.error("Script stderr:", result.stderr);
        throw new Error(`Script failed with exit code ${result.exitCode}`);
      }

      // Read the embedding from the output file
      const outputContent = await workspace.fs.readFile(
        "/home/daytona/output.json"
      );
      const embedding = JSON.parse(outputContent);

      if (!Array.isArray(embedding)) {
        throw new Error(`Unexpected embedding format: ${typeof embedding}`);
      }

      // Log embedding details
      console.log(`Embedding for ${review.review_url}:`);
      console.log(`- Type: ${typeof embedding}`);
      console.log(`- Length: ${embedding.length}`);
      console.log(
        `- First few values: ${embedding
          .slice(0, 5)
          .map((v) => v.toFixed(4))
          .join(", ")}`
      );
      console.log(`- Is array: ${Array.isArray(embedding)}`);
      console.log(
        `- All numbers: ${embedding.every((val) => typeof val === "number")}`
      );

      // Verify embedding dimensions
      if (embedding.length !== 384) {
        throw new Error(
          `Invalid embedding size: ${embedding.length}, expected 384`
        );
      }

      // Prepare payload with all available fields
      const payload = {
        review_url: review.review_url,
        title: review.title || "",
        artists: review.artists
          ? review.artists.split(",").map((a) => a.trim())
          : [],
        authors: review.authors
          ? review.authors.split(",").map((a) => a.trim())
          : [],
        genres: review.genres
          ? review.genres.split(",").map((g) => g.trim())
          : [],
        labels: review.labels
          ? review.labels.split(",").map((l) => l.trim())
          : [],
        score: review.score,
        best_new_music: review.best_new_music === 1,
        best_new_reissue: review.best_new_reissue === 1,
        pub_date: review.pub_date,
        release_years: review.release_years
          ? review.release_years.split(",").map(Number)
          : [],
        body: review.body,
      };

      // Generate a UUID for the point ID
      const id = uuidv4();

      // Upsert to Qdrant with properly formatted vector
      await qdrant.upsert(COLLECTION_NAME, {
        points: [
          {
            id,
            vector: {
              data: embedding,
            },
            payload: payload,
          },
        ],
      });

      console.log(`Successfully processed review ${review.review_url}`);
    } catch (error) {
      console.error(`Error processing review ${review.review_url}:`, error);
    }
  }
}

async function processReviews() {
  // Open SQLite database
  const db = await open({
    filename: "data.sqlite3",
    driver: sqlite3.Database,
  });

  // Get all reviews with their related data
  const reviews = await db.all(`
    SELECT 
      r.review_url,
      r.pub_date,
      r.body,
      t.title,
      t.score,
      t.best_new_music,
      t.best_new_reissue,
      GROUP_CONCAT(DISTINCT a.name) as artists,
      GROUP_CONCAT(DISTINCT g.genre) as genres,
      GROUP_CONCAT(DISTINCT l.label) as labels,
      GROUP_CONCAT(DISTINCT auth.author) as authors,
      GROUP_CONCAT(DISTINCT try.release_year) as release_years
    FROM reviews r
    LEFT JOIN tombstones t ON r.review_url = t.review_url
    LEFT JOIN artist_review_map arm ON r.review_url = arm.review_url
    LEFT JOIN artists a ON arm.artist_id = a.artist_id
    LEFT JOIN genre_review_map grm ON r.review_url = grm.review_url
    LEFT JOIN genre_review_map g ON grm.review_url = g.review_url
    LEFT JOIN tombstone_label_map tlm ON t.review_tombstone_id = tlm.review_tombstone_id
    LEFT JOIN tombstone_label_map l ON tlm.review_tombstone_id = l.review_tombstone_id
    LEFT JOIN author_review_map auth ON r.review_url = auth.review_url
    LEFT JOIN tombstone_release_year_map try ON t.review_tombstone_id = try.review_tombstone_id
    WHERE r.is_standard_review = 1
    GROUP BY r.review_url
    LIMIT 5  -- Just process 5 reviews for testing
  `);

  await db.close();

  // Create a single workspace for testing
  const daytona = new Daytona({
    apiKey: process.env.DAYTONA_API_KEY,
    serverUrl: "https://app.daytona.io/api",
    target: "us",
  });

  try {
    const workspace = await daytona.create({ language: "typescript" });
    console.log(`Created workspace for processing ${reviews.length} reviews`);
    await processReviewChunk(reviews, workspace);
    await workspace.delete();
  } catch (error) {
    console.error("Error in workspace:", error);
  }
}

async function main() {
  try {
    await setupQdrant();
    await processReviews();
    console.log("Finished processing all reviews");
  } catch (error) {
    console.error("Error:", error);
  }
}

main();

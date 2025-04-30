import { pipeline } from "@xenova/transformers";
import { QdrantClient } from "@qdrant/js-client-rest";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import fs from "fs";
import dotenv from "dotenv";
import { v4 as uuidv4 } from "uuid";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables
dotenv.config();

// Verify environment variables
const qdrantUrl = process.env.QDRANT_URL;
const qdrantApiKey = process.env.QDRANT_API_KEY;

if (!qdrantUrl || !qdrantUrl.startsWith("http")) {
  throw new Error("QDRANT_URL must be set and start with http:// or https://");
}

if (!qdrantApiKey) {
  throw new Error("QDRANT_API_KEY must be set");
}

console.log("Qdrant URL:", qdrantUrl);
console.log("Qdrant API Key:", qdrantApiKey ? "***" : "not set");

// Read reviews from JSON file
const reviews = JSON.parse(
  fs.readFileSync(join(__dirname, "reviews.json"), "utf8")
);

// Initialize Qdrant client
const qdrant = new QdrantClient({
  url: qdrantUrl,
  apiKey: qdrantApiKey,
});

// Initialize the embedding pipeline
const embedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

// Create collection if it doesn't exist
await qdrant.createCollection("reviews", {
  vectors: {
    size: 384,
    distance: "Cosine",
  },
});

console.log(`Processing ${reviews.length} reviews...`);

// Process each review
for (const review of reviews) {
  try {
    // Log the review being processed
    console.log("Processing review:", {
      url: review.review_url,
      title: review.title,
      bodyLength: review.body?.length || 0,
    });

    // Generate embedding
    const embedding = await embedder(review.body, {
      pooling: "mean",
      normalize: true,
    });

    // Log the raw embedding output for debugging
    console.log("Raw embedding output:", {
      type: typeof embedding,
      isArray: Array.isArray(embedding),
      keys: embedding ? Object.keys(embedding) : null,
      dataType: embedding?.data ? typeof embedding.data : null,
      dataIsArray: embedding?.data ? Array.isArray(embedding.data) : false,
      dataLength: embedding?.data ? Object.keys(embedding.data).length : null,
      dims: embedding?.dims,
      size: embedding?.size,
    });

    // Extract and convert the vector data
    let vectorData;
    if (embedding && embedding.data) {
      // Convert the object with numeric keys into an array
      const dataObj = embedding.data;
      vectorData = new Array(384);
      for (let i = 0; i < 384; i++) {
        vectorData[i] = dataObj[i];
      }
    }

    // Log the extracted vector data
    console.log("Extracted vector data:", {
      isArray: Array.isArray(vectorData),
      type: typeof vectorData,
      length: vectorData?.length,
      sample: vectorData ? JSON.stringify(vectorData.slice(0, 5)) : null,
    });

    if (!vectorData || !Array.isArray(vectorData)) {
      throw new Error(
        `Could not extract vector data from embedding. Embedding structure: ${JSON.stringify(
          embedding,
          null,
          2
        )}`
      );
    }

    // Validate vector data
    if (vectorData.length !== 384) {
      throw new Error(
        `Invalid vector size: ${vectorData.length}, expected 384`
      );
    }

    // Generate UUID for the point
    const pointId = uuidv4();

    // Prepare the point data
    const point = {
      id: pointId,
      vector: vectorData,
      payload: {
        review_url: review.review_url,
        title: review.title,
        artists: review.artists,
        artist_count: review.artist_count,
        authors: review.authors,
        genres: review.genres,
        body: review.body,
        labels: review.labels,
        score: review.score,
        best_new_music: review.best_new_music,
        pub_date: review.pub_date,
        release_year: review.release_year,
      },
    };

    // Log the point data for debugging
    console.log("Point data:", {
      id: point.id,
      vectorLength: point.vector.length,
      vectorType: typeof point.vector,
      vectorIsArray: Array.isArray(point.vector),
      vectorSample: JSON.stringify(point.vector.slice(0, 5)),
      payload: point.payload,
    });

    // Upsert to Qdrant
    await qdrant.upsert("reviews", {
      points: [point],
    });

    console.log(
      `Successfully processed review ${review.review_url} with ID ${pointId}`
    );
  } catch (error) {
    console.error(`Error processing review ${review.review_url}:`, error);
    // Log the full error details
    console.error("Error details:", {
      message: error.message,
      stack: error.stack,
      data: error.data,
      review: {
        url: review.review_url,
        title: review.title,
        bodyLength: review.body?.length || 0,
      },
    });
  }
}

console.log("Processing complete!");

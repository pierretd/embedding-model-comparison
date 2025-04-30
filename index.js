import { Daytona } from "@daytonaio/sdk";
import fs from "fs";
import path from "path";
import Database from "better-sqlite3";
import { v4 as uuidv4 } from "uuid";

// Helper function to retry operations
async function retry(operation, maxRetries = 3, delay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      console.log(`Attempt ${i + 1} failed, retrying in ${delay}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
}

// Initialize the Daytona client
const daytona = new Daytona({
  apiKey:
    "dtn_ad3e1a6a39ebe3a24061ab95112fbd487051d588443e857e5cc8798fc6d1d3c3",
  serverUrl: "https://app.daytona.io/api",
  target: "us",
});

// Create a workspace with JavaScript support
const workspace = await daytona.create({ language: "javascript" });
console.log("Workspace ID:", workspace.id);

// Using interactive sessions
const sessionId = "my-session";
await retry(() => workspace.process.createSession(sessionId));
console.log("Session created successfully");

try {
  // Read reviews from local database
  console.log("Reading reviews from local database...");
  const db = new Database(path.join(process.cwd(), "data.sqlite3"));
  const reviews = db
    .prepare(
      `
    SELECT 
      review_url,
      artist_count,
      artists,
      title,
      score,
      best_new_music,
      authors,
      genres,
      labels,
      pub_date,
      release_year,
      body
    FROM standard_reviews_flat 
    LIMIT 5
  `
    )
    .all();
  console.log(`Found ${reviews.length} reviews to process`);

  // Change to /home/daytona directory
  console.log("Changing to /home/daytona directory...");
  await retry(() =>
    workspace.process.executeSessionCommand(sessionId, {
      command: "cd /home/daytona",
    })
  );

  // Create package.json with required dependencies
  console.log("Creating package.json...");
  const packageJson = {
    name: "review-embedding",
    version: "1.0.0",
    type: "module",
    dependencies: {
      "@xenova/transformers": "^2.15.0",
      "@qdrant/js-client-rest": "^1.13.0",
      dotenv: "^16.4.1",
      uuid: "^9.0.1",
    },
  };

  const packageFile = new File(
    [JSON.stringify(packageJson, null, 2)],
    "package.json",
    { type: "text/plain" }
  );
  await retry(() =>
    workspace.fs.uploadFile("/home/daytona/package.json", packageFile)
  );
  console.log("package.json uploaded successfully");

  // Create reviews.json file
  console.log("Creating reviews.json...");
  const reviewsFile = new File(
    [JSON.stringify(reviews, null, 2)],
    "reviews.json",
    { type: "application/json" }
  );
  await retry(() =>
    workspace.fs.uploadFile("/home/daytona/reviews.json", reviewsFile)
  );
  console.log("reviews.json uploaded successfully");

  // Upload the embedding script
  console.log("Uploading embedding script...");
  const embedScriptContent = fs.readFileSync(
    path.join(process.cwd(), "embed.js"),
    "utf8"
  );
  const embedScriptFile = new File([embedScriptContent], "embed.js", {
    type: "text/plain",
  });
  await retry(() =>
    workspace.fs.uploadFile("/home/daytona/embed.js", embedScriptFile)
  );
  console.log("embed.js uploaded successfully");

  // Create .env file
  console.log("Creating .env file...");
  const envContent = `QDRANT_URL=${process.env.QDRANT_URL}
QDRANT_API_KEY=${process.env.QDRANT_API_KEY}`;
  const envFile = new File([envContent], ".env", { type: "text/plain" });
  await retry(() => workspace.fs.uploadFile("/home/daytona/.env", envFile));
  console.log(".env file uploaded successfully");

  // Install dependencies
  console.log("Installing dependencies...");
  const installResponse = await retry(() =>
    workspace.process.executeSessionCommand(sessionId, {
      command: "npm install",
    })
  );
  console.log("Installation output:", installResponse.stdout);

  // Run the embedding script
  console.log("Running embedding script...");
  const runResponse = await retry(() =>
    workspace.process.executeSessionCommand(sessionId, {
      command: "node embed.js",
    })
  );
  if (runResponse.stderr) {
    console.log("Script errors:", runResponse.stderr);
  }
} catch (error) {
  console.error("Error:", error);
} finally {
  // Clean up
  try {
    await retry(() => workspace.process.deleteSession(sessionId));
    await retry(() => daytona.remove(workspace));
  } catch (error) {
    console.error("Error during cleanup:", error);
  }
}

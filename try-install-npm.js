import { Daytona } from "@daytonaio/sdk";
import dotenv from "dotenv";

dotenv.config();

// Initialize the Daytona client
const daytona = new Daytona({
  apiKey: process.env.DAYTONA_API_KEY,
  serverUrl: "https://app.daytona.io/api",
  target: "us",
});

async function main() {
  let workspace;
  let sessionId;
  try {
    // Create a new workspace with Node.js support
    console.log("Creating new workspace...");
    workspace = await daytona.create({ language: "javascript" });
    console.log("Workspace created successfully");

    // Create a session for running commands
    sessionId = "npm-test";
    await workspace.process.createSession(sessionId);
    console.log("Session created successfully");

    // Create package.json
    const packageJson = {
      name: "lodash-test",
      version: "1.0.0",
      type: "module",
      dependencies: {
        lodash: "^4.17.21",
      },
    };

    const packageFile = new File(
      [JSON.stringify(packageJson, null, 2)],
      "package.json",
      { type: "text/plain" }
    );
    await workspace.fs.uploadFile("/home/daytona/package.json", packageFile);
    console.log("package.json created");

    // Change to the workspace directory
    await workspace.process.executeSessionCommand(sessionId, {
      command: "cd /home/daytona",
    });

    // Install lodash
    console.log("Installing lodash...");
    const installResponse = await workspace.process.executeSessionCommand(
      sessionId,
      {
        command: "npm install",
      }
    );
    console.log("Installation output:", installResponse);

    // List installed packages
    console.log("\nListing installed packages...");
    const listResponse = await workspace.process.executeSessionCommand(
      sessionId,
      {
        command: "npm list",
      }
    );
    console.log("Installed packages:", listResponse);
  } catch (error) {
    console.error("Error:", error);
  } finally {
    // Clean up
    try {
      if (workspace) {
        if (sessionId) {
          await workspace.process.deleteSession(sessionId);
        }
        await daytona.remove(workspace);
      }
    } catch (error) {
      console.error("Error during cleanup:", error);
    }
  }
}

main();

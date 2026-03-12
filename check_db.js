require("dotenv").config();
const { Pinecone } = require("@pinecone-database/pinecone");
const { pipeline } = require("@xenova/transformers");

// ================= CONFIG =================
const NAMESPACE = "default";
const TOP_K = 2000; // Fetch 2000 chunks to get a good sample of files

// ================= SERVICES =================
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pinecone.index(process.env.PINECONE_INDEX);

async function main() {
  console.log("------------------------------------------------");
  console.log("🔍 AUDITING PINECONE DATABASE...");
  console.log("------------------------------------------------");

  // 1. Load Model to generate a dummy search vector
  console.log("🧠 Loading embedding model to probe DB...");
  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  
  // Create a generic vector for "Law Act Section" to cast a wide net
  const output = await embedder("Law Act Section India Penalty Rule", { pooling: "mean", normalize: true });
  const queryVector = Array.from(output.data).map(Number);

  console.log("📡 Querying Database for file sources...");

  try {
    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: TOP_K, 
      includeMetadata: true,
    });

    const matches = searchResult.matches || [];

    if (matches.length === 0) {
        console.log("❌ Database is EMPTY. No vectors found.");
        return;
    }

    // 2. Extract Unique Source Names
    const filesFound = new Set();
    matches.forEach(m => {
        if (m.metadata && m.metadata.source) {
            filesFound.add(m.metadata.source);
        }
    });

    console.log(`\n✅ FOUND ${matches.length} CHUNKS FROM ${filesFound.size} UNIQUE FILES:\n`);
    
    // 3. Print List
    const sortedFiles = Array.from(filesFound).sort();
    sortedFiles.forEach((file, index) => {
        console.log(`${index + 1}. ${file}`);
    });

    console.log("\n------------------------------------------------");
    console.log("Note: This is a sample based on the top 2000 vectors.");
    console.log("If a file is missing here but was ingested, it might be deeper in the DB.");
    console.log("------------------------------------------------");

  } catch (error) {
    console.error("❌ Error auditing database:", error);
  }
}

main();
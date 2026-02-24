const fs = require("fs");
const path = require("path");
require("dotenv").config();

let pdfParse = require("pdf-parse");
if (typeof pdfParse !== "function" && pdfParse.default) {
  pdfParse = pdfParse.default;
}

const { Pinecone } = require("@pinecone-database/pinecone");
const { pipeline } = require("@xenova/transformers");

// ================= CONFIG =================
const CHUNK_SIZE = 1000;
const OVERLAP = 200;
const BATCH_SIZE = 20;
const NAMESPACE = "default";

// ================= INIT =================
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX);

// ================= LOAD LOCAL EMBEDDING MODEL =================
let embedder;

async function loadModel() {
  console.log("🧠 Loading local embedding model...");
  embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  console.log("✅ Model loaded.");
}

// ================= GET EMBEDDING =================
async function getEmbedding(text) {
  const output = await embedder(text, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data);
}

// ================= UPSERT =================
async function safeUpsert(records) {
  if (!records.length) return;

  await index.upsert({
    namespace: NAMESPACE,
    records: records,
  });
}

// ================= MAIN =================
async function main() {
  console.log("------------------------------------------------");
  console.log("🚀 STARTING LAWSPHERE INGESTION (LOCAL EMBEDDINGS)");
  console.log("------------------------------------------------");

  await loadModel();

  const filePath = path.join(__dirname, "data", "bns_book.pdf");

  if (!fs.existsSync(filePath)) {
    console.error("❌ PDF NOT FOUND");
    return;
  }

  const dataBuffer = fs.readFileSync(filePath);
  const pdfData = await pdfParse(dataBuffer);
  let rawText = pdfData.text.replace(/\s+/g, " ").trim();

  console.log(`📄 PDF Loaded. Characters: ${rawText.length}`);

  let chunks = [];

  for (let i = 0; i < rawText.length; i += (CHUNK_SIZE - OVERLAP)) {
    const chunk = rawText.substring(i, i + CHUNK_SIZE);
    if (chunk.length > 100) chunks.push(chunk);
  }

  console.log(`🧩 Created ${chunks.length} chunks.`);
  console.log("⚡ Generating LOCAL embeddings & uploading...\n");

  let batch = [];
  let total = 0;

  for (let i = 0; i < chunks.length; i++) {
    const embedding = await getEmbedding(chunks[i]);

    batch.push({
      id: `chunk-${i}`,
      values: embedding,
      metadata: {
        text: chunks[i],
        source: "BNS PDF"
      }
    });

    process.stdout.write(".");

    if (batch.length >= BATCH_SIZE) {
      await safeUpsert(batch);
      total += batch.length;
      batch = [];
    }
  }

  if (batch.length > 0) {
    await safeUpsert(batch);
    total += batch.length;
  }

  console.log(`\n\n✅ SUCCESS: Uploaded ${total} chunks.`);
}

main();

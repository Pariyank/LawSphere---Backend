const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Fix PDF Library Import
let pdfParse = require('pdf-parse');
if (typeof pdfParse !== 'function' && pdfParse.default) {
    pdfParse = pdfParse.default;
}

const { Pinecone } = require('@pinecone-database/pinecone');
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { pipeline } = require("@xenova/transformers");

// ================= CONFIG =================
const DATA_DIR = path.join(__dirname, 'data');
const NAMESPACE = "default";
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;
const BATCH_SIZE = 50; 
const INDEX_NAME = process.env.PINECONE_INDEX || "lawsphere-index";

// ================= SERVICES =================
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

let embedder = null;
let indexHost = ""; // We will fetch this dynamically

async function loadModel() {
  console.log("🧠 Loading local embedding model (Xenova)...");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log("✅ Model loaded.");
}

async function getEmbedding(text) {
  if (!embedder) await loadModel();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  
  // Force convert to plain numbers to avoid SDK issues
  const rawArray = Array.from(output.data).map(n => Number(n));
  
  // Sanity check for NaN
  if (rawArray.some(isNaN)) {
      throw new Error("Embedding produced NaN values.");
  }
  return rawArray;
}

// Clean string for ID (Alphanumeric only)
function cleanString(str) {
    return str.replace(/[^a-zA-Z0-9]/g, '');
}

// Clean text for Metadata (Remove null bytes that crash Pinecone)
function sanitizeText(str) {
    return str.replace(/\0/g, '').trim();
}

// 🟢 DIRECT HTTP UPSERT (Bypasses SDK Validation Bugs)
async function directHttpUpsert(vectors) {
    if (!indexHost) throw new Error("Index Host not initialized.");

    const url = `https://${indexHost}/vectors/upsert`;
    
    const payload = {
        vectors: vectors,
        namespace: NAMESPACE
    };

    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Api-Key": process.env.PINECONE_API_KEY,
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const errText = await response.text();
        throw new Error(`HTTP Error ${response.status}: ${errText}`);
    }

    const json = await response.json();
    return json.upsertedCount;
}

async function processFile(fileName) {
    const filePath = path.join(DATA_DIR, fileName);
    console.log(`\n📄 Processing: ${fileName}...`);

    try {
        const dataBuffer = fs.readFileSync(filePath);
        const pdfData = await pdfParse(dataBuffer);
        const rawText = sanitizeText(pdfData.text.replace(/\s+/g, " "));

        if (rawText.length < 100) {
            console.log(`⚠️ Skipped ${fileName} (Content empty/scanned).`);
            return 0;
        }

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: CHUNK_SIZE,
            chunkOverlap: CHUNK_OVERLAP,
            separators: ["\n\n", "Section", "Article", "\n", ". ", " "] 
        });

        const outputDocuments = await splitter.createDocuments([rawText]);
        const chunks = outputDocuments.map(doc => doc.pageContent);

        console.log(`   🧩 Split into ${chunks.length} chunks.`);

        let vectors = [];
        let count = 0;
        const readableSource = fileName.replace('.pdf', '').replace(/_/g, ' ');
        const idBase = cleanString(fileName);

        for (let i = 0; i < chunks.length; i++) {
            const chunkText = sanitizeText(chunks[i]);
            
            const sectionMatch = chunkText.match(/(Section|Article)\s+(\d+[A-Z]*)/i);
            const sectionLabel = sectionMatch ? `${sectionMatch[1]} ${sectionMatch[2]}` : "General";

            try {
                const embedding = await getEmbedding(`[Law: ${readableSource}] ${chunkText}`);
                
                vectors.push({
                    id: `${idBase}-${i}`,
                    values: embedding,
                    metadata: {
                        text: chunkText,
                        section: sectionLabel,
                        source: readableSource
                    }
                });
            } catch (embedErr) {
                console.error(`Skipping chunk ${i} due to embedding error.`);
            }

            process.stdout.write(".");

            if (vectors.length >= BATCH_SIZE) {
                try {
                    await directHttpUpsert(vectors);
                    count += vectors.length;
                    vectors = [];
                } catch (e) {
                    console.error(`\n❌ Batch Error: ${e.message}`);
                    vectors = [];
                }
            }
        }

        // Final Batch
        if (vectors.length > 0) {
            try {
                await directHttpUpsert(vectors);
                count += vectors.length;
            } catch (e) {
                console.error(`\n❌ Final Batch Error: ${e.message}`);
            }
        }

        console.log(`\n   ✅ Uploaded ${count} vectors.`);
        return count;

    } catch (e) {
        console.error(`\n❌ File Error ${fileName}:`, e.message);
        return 0;
    }
}

async function main() {
    console.log("------------------------------------------------");
    console.log("🚀 STARTING UNIVERSAL INGESTION (DIRECT HTTP)");
    console.log("------------------------------------------------");

    if (!fs.existsSync(DATA_DIR)) {
        console.error("❌ 'data' directory missing!");
        return;
    }

    // 🟢 1. GET INDEX HOST
    try {
        console.log("🔌 Connecting to Pinecone...");
        const indexDescription = await pinecone.describeIndex(INDEX_NAME);
        indexHost = indexDescription.host;
        console.log(`✅ Connected to Host: ${indexHost}`);
    } catch (e) {
        console.error("❌ Could not find Index. Make sure 'lawsphere-index' exists on Pinecone Console.");
        console.error(e);
        return;
    }

    await loadModel();

    const files = fs.readdirSync(DATA_DIR).filter(file => file.toLowerCase().endsWith('.pdf'));
    console.log(`📚 Found ${files.length} Legal Documents.`);

    let totalVectors = 0;
    for (const file of files) {
        totalVectors += await processFile(file);
    }

    console.log("------------------------------------------------");
    console.log(`🎉 GRAND TOTAL: ${totalVectors} chunks uploaded.`);
}

main();
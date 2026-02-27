const fs = require('fs');
const path = require('path');
require('dotenv').config();

let pdfParse = require('pdf-parse');
if (typeof pdfParse !== 'function' && pdfParse.default) {
    pdfParse = pdfParse.default;
}

const { Pinecone } = require('@pinecone-database/pinecone');
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { pipeline } = require("@xenova/transformers");

const DATA_DIR = path.join(__dirname, 'data');
const NAMESPACE = "default";
const CHUNK_SIZE = 800;
const CHUNK_OVERLAP = 150;
const BATCH_SIZE = 50;

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pinecone.index(process.env.PINECONE_INDEX);

let embedder = null;

async function loadModel() {
  console.log("🧠 Loading local embedding model (Xenova)...");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log("✅ Model loaded.");
}

async function getEmbedding(text) {
  if (!embedder) await loadModel();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

async function processFile(fileName) {
    const filePath = path.join(DATA_DIR, fileName);
    console.log(`\n📄 Processing: ${fileName}...`);

    try {
        const dataBuffer = fs.readFileSync(filePath);
        const pdfData = await pdfParse(dataBuffer);
    
        const rawText = pdfData.text.replace(/\s+/g, " ").trim();

        if (rawText.length < 100) {
            console.log(`⚠️ Skipped ${fileName} (Empty)`);
            return 0;
        }

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: CHUNK_SIZE,
            chunkOverlap: CHUNK_OVERLAP,
            separators: ["\n\n", "Section", "Article", "CHAPTER", "\n", ". "] 
        });

        const outputDocuments = await splitter.createDocuments([rawText]);
        const chunks = outputDocuments.map(doc => doc.pageContent);

        console.log(`   🧩 Split into ${chunks.length} chunks.`);

        let vectors = [];
        let count = 0;
        

        const sourceName = fileName.replace('.pdf', '').toUpperCase(); 

        for (let i = 0; i < chunks.length; i++) {
            const chunkText = chunks[i];
       
            const sectionMatch = chunkText.match(/(Section|Article)\s+(\d+[A-Z]*)/i);
            const sectionLabel = sectionMatch ? `${sectionMatch[1]} ${sectionMatch[2]}` : "General";

         
            const textToEmbed = `[Law: ${sourceName}] ${chunkText}`;

            const embedding = await getEmbedding(textToEmbed);

            vectors.push({
                id: `${sourceName}-${i}`,
                values: embedding,
                metadata: {
                    text: chunkText, 
                    section: sectionLabel,
                    source: sourceName 
                }
            });

            process.stdout.write(".");

            if (vectors.length >= BATCH_SIZE) {
                await index.namespace(NAMESPACE).upsert(vectors);
                count += vectors.length;
                vectors = [];
            }
        }

        if (vectors.length > 0) {
            await index.namespace(NAMESPACE).upsert(vectors);
            count += vectors.length;
        }

        console.log(`\n   ✅ Uploaded ${count} vectors for ${sourceName}`);
        return count;

    } catch (e) {
        console.error(`\n❌ Error processing ${fileName}:`, e.message);
        return 0;
    }
}

async function main() {
    console.log("------------------------------------------------");
    console.log("🚀 STARTING UNIVERSAL LAW INGESTION");
    console.log("------------------------------------------------");

    if (!fs.existsSync(DATA_DIR)) {
        console.error("❌ 'data' directory missing!");
        return;
    }

    await loadModel();

    const files = fs.readdirSync(DATA_DIR).filter(file => file.toLowerCase().endsWith('.pdf'));

    if (files.length === 0) {
        console.error("❌ No PDF files found in 'data' folder!");
        return;
    }

    console.log(`📚 Found ${files.length} Documents:`, files);

    let totalVectors = 0;
    for (const file of files) {
        totalVectors += await processFile(file);
    }

    console.log("------------------------------------------------");
    console.log(`🎉 GRAND TOTAL: ${totalVectors} chunks uploaded.`);
    console.log("🧠 LawSphere Brain is updated.");
    console.log("------------------------------------------------");
}

main();

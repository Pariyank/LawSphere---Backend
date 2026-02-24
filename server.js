require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pinecone } = require("@pinecone-database/pinecone");
const { pipeline } = require("@xenova/transformers");
const Groq = require("groq-sdk");

const app = express();
app.use(cors());
app.use(express.json());

// ================= CONFIG =================
const PORT = 3000;
const NAMESPACE = "default";
// Use the latest versatile Llama model
const LLM_MODEL = "llama-3.3-70b-versatile"; 

// ================= SERVICES =================
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pinecone.index(process.env.PINECONE_INDEX);

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});

// ================= LOCAL EMBEDDING LOGIC =================
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

// ================= ROUTER =================
const router = express.Router();

// 1. ASK ROUTE (With Language Support)
router.post("/ask", async (req, res) => {
  try {
    // 🟢 1. Accept 'language'
    const { query, language } = req.body;
    console.log(`📩 Query: ${query} | Lang: ${language}`);

    if (!query) return res.status(400).json({ error: "Query required" });

    // 🟢 2. Define Language Rule
    const langInstruction = language === "hindi" 
        ? "CRITICAL RULE: Answer the user's question entirely in HINDI (Devanagari script). Use simple, clear legal Hindi." 
        : "Answer in English.";

    // A. Embed
    const queryVector = await getEmbedding(query);

    // B. Search
    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 5,
      includeMetadata: true,
    });

    const matches = searchResult.matches || [];
    
    // C. Context
    const context = matches
      .map((m, i) => `Source ${i + 1}:\n${m.metadata?.text || ""}`)
      .join("\n\n");

    // D. Generate
    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                // 🟢 3. Inject Language Instruction
                content: `You are LawSphere, an expert legal AI for Bharatiya Nyaya Sanhita (BNS). 
                
                ${langInstruction}

                STRICT RULES:
                1. Answer ONLY using the provided context.
                2. Cite relevant Section numbers.
                3. Format in Markdown.`
            },
            {
                role: "user",
                content: `Context:\n${context}\n\nQuestion:\n${query}`
            }
        ],
        model: LLM_MODEL,
        temperature: 0.1, 
    });

    const answer = completion.choices[0]?.message?.content || "No answer generated.";

    console.log("✅ Answer sent.");

    res.json({
      formattedAnswer: answer,
      reasoning: "Vector Search",
      semanticTags: ["BNS", "Legal"],
      retrievedSources: matches.map((m, i) => ({
        sourceNumber: i + 1,
        snippet: (m.metadata?.text || "").substring(0, 200) + "..."
      }))
    });

  } catch (error) {
    console.error("❌ Chat Error:", error);
    res.status(500).json({ formattedAnswer: "Server Error: " + error.message, retrievedSources: [] });
  }
});

// 2. COMPARE ROUTE
router.post("/compare", async (req, res) => {
  try {
    const { section1, section2 } = req.body;
    console.log(`⚖️ Comparing: ${section1} vs ${section2}`);

    if (!section1 || !section2) {
      return res.status(400).json({ error: "Both sections required" });
    }

    // Embed BOTH
    const vec1 = await getEmbedding(section1);
    const vec2 = await getEmbedding(section2);

    // Search BOTH
    const [result1, result2] = await Promise.all([
        index.namespace(NAMESPACE).query({ vector: vec1, topK: 3, includeMetadata: true }),
        index.namespace(NAMESPACE).query({ vector: vec2, topK: 3, includeMetadata: true })
    ]);

    const matches = [...(result1.matches || []), ...(result2.matches || [])];
    const uniqueContext = Array.from(new Set(matches.map(m => m.metadata?.text))).join("\n\n---\n\n");

    if (!uniqueContext || uniqueContext.length < 50) {
        return res.json({ formattedAnswer: "I could not find these sections in the uploaded BNS PDF.", retrievedSources: [] });
    }

    // Generate Comparison
    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are a strict legal expert for the Bharatiya Nyaya Sanhita (BNS) of India.
                CRITICAL RULES:
                1. Use ONLY the provided CONTEXT.
                2. Do NOT use outside knowledge (IPC/Bangladesh).
                3. Output a Markdown Table.`
            },
            {
                role: "user",
                content: `
                CONTEXT FROM PDF:
                ${uniqueContext}

                TASK: Compare "${section1}" and "${section2}".
                Rows: Definition, Punishment, Cognizable status.
                `
            }
        ],
        model: LLM_MODEL,
        temperature: 0.1,
    });

    const answer = completion.choices[0]?.message?.content || "Comparison failed.";

    res.json({
      formattedAnswer: answer,
      reasoning: "Comparison based on BNS PDF",
      semanticTags: ["Compare", "BNS"],
      retrievedSources: []
    });

  } catch (error) {
    console.error("❌ Compare Error:", error);
    res.status(500).json({ formattedAnswer: "Comparison Error: " + error.message, retrievedSources: [] });
  }
});

app.use("/api", router);

app.listen(PORT, "0.0.0.0", async () => {
  await loadModel();
  console.log(`🚀 LawSphere Backend running on port ${PORT}`);
});
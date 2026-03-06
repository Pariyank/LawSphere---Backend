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
const PORT = process.env.PORT || 3000;
const NAMESPACE = "default";
const LLM_MODEL = "llama-3.3-70b-versatile"; 

// ================= SERVICES =================
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX);
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ================= LOCAL EMBEDDING =================
let embedder = null;

async function loadModel() {
  console.log("🧠 Loading local embedding model (Xenova)...");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log("✅ Model loaded.");
}

async function getEmbedding(text) {
  if (!embedder) await loadModel();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data).map(Number);
}

// ================= ROUTER =================
const router = express.Router();

router.get("/", (req, res) => res.send("🚀 LawSphere Brain is Active"));

// ---------------------------------------------------------
// 1. CHAT ROUTE (Universal Legal Search)
// ---------------------------------------------------------
router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`📩 Query: "${query}"`);

    if (!query) return res.status(400).json({ error: "Query required" });

    // 1. Embed Query
    const queryVector = await getEmbedding(query);

    // 2. Vector Search (Increased Scope)
    // 🟢 CHANGED: topK increased to 20. 
    // With 51 PDFs, we need to cast a wider net to find the specific Act.
    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 20, 
      includeMetadata: true,
    });

    const matches = searchResult.matches || [];

    // 3. Build Context with STRONG Source Headers
    const context = matches
      .map((m, i) => `DOCUMENT: ${m.metadata?.source?.toUpperCase() || "UNKNOWN"}\nCONTENT: ${m.metadata?.text || ""}`)
      .join("\n\n====================\n\n");

    // 4. Language Instruction
    const langInstruction = language === "hindi" 
        ? "OUTPUT RULE: Answer in HINDI (Devanagari)." 
        : "OUTPUT RULE: Answer in English.";

    // 5. Universal Prompt (Removes BNS Bias)
    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are LawSphere, a Database Search Engine for Indian Laws.
                
                You have access to 50+ different Legal Documents (BNS, BNSS, IT Act, Wages Act, Contracts Act, etc.).
                
                ${langInstruction}

                CRITICAL INSTRUCTIONS:
                1. Your Goal: Find the answer to the user's question from the provided 'CONTEXT'.
                2. **SOURCE CHECK:** Look at the "DOCUMENT:" header above each chunk. 
                3. **RELEVANCE:** If the user asks about "Act 4 of 1936", ignore chunks labeled "BNS" or "IPC". Only use chunks that look like "Payment of Wages Act" or similar.
                4. **NO HALLUCINATION:** If the exact answer is not in the context, say: "I could not find details about this specific Act in the database." Do NOT default to BNS.
                5. **CITATION:** Always start your answer with: "According to [Insert Document Name]..."
                6. Format in Markdown.`
            },
            {
                role: "user",
                content: `CONTEXT:\n${context}\n\nUSER QUESTION:\n${query}`
            }
        ],
        model: LLM_MODEL,
        temperature: 0.1, 
    });

    const answer = completion.choices[0]?.message?.content || "No answer generated.";

    console.log("✅ Chat Answer Sent.");

    res.json({
      formattedAnswer: answer,
      reasoning: "Universal DB Search",
      semanticTags: matches.slice(0, 3).map(m => m.metadata?.source || "Law"),
      retrievedSources: matches.slice(0, 5).map((m, i) => ({
        sourceNumber: i + 1,
        snippet: `[${m.metadata?.source}] ${m.metadata?.text?.substring(0, 100)}...`
      }))
    });

  } catch (error) {
    console.error("❌ Chat Error:", error);
    res.status(500).json({ formattedAnswer: "Server Error: " + error.message, retrievedSources: [] });
  }
});

// ---------------------------------------------------------
// 2. COMPARE ROUTE
// ---------------------------------------------------------
router.post("/compare", async (req, res) => {
  try {
    const { section1, section2 } = req.body;
    
    const vec1 = await getEmbedding(section1);
    const vec2 = await getEmbedding(section2);

    const [result1, result2] = await Promise.all([
        index.namespace(NAMESPACE).query({ vector: vec1, topK: 5, includeMetadata: true }),
        index.namespace(NAMESPACE).query({ vector: vec2, topK: 5, includeMetadata: true })
    ]);

    const matches = [...(result1.matches || []), ...(result2.matches || [])];
    const uniqueContext = Array.from(new Set(matches.map(m => `[Doc: ${m.metadata.source}] ${m.metadata.text}`))).join("\n\n");

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `Compare the two requested topics using ONLY the provided Context. Output a Markdown Table.`
            },
            {
                role: "user",
                content: `CONTEXT: ${uniqueContext}\nTASK: Compare "${section1}" and "${section2}".`
            }
        ],
        model: LLM_MODEL,
        temperature: 0.1,
    });

    res.json({
      formattedAnswer: completion.choices[0]?.message?.content || "Comparison failed.",
      semanticTags: ["Comparison"],
      retrievedSources: []
    });

  } catch (error) {
    res.status(500).json({ formattedAnswer: "Error: " + error.message, retrievedSources: [] });
  }
});


app.use("/api", router);

app.listen(PORT, "0.0.0.0", async () => {
  await loadModel();
  console.log(`🚀 LawSphere Backend running on port ${PORT}`);
});
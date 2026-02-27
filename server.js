require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pinecone } = require("@pinecone-database/pinecone");
const { pipeline } = require("@xenova/transformers");
const Groq = require("groq-sdk");

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3000;
const NAMESPACE = "default";
const LLM_MODEL = "llama-3.3-70b-versatile"; 

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pinecone.index(process.env.PINECONE_INDEX);

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});

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

const router = express.Router();

// 1. ASK ROUTE
router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`📩 Query: ${query} | Lang: ${language}`);

    if (!query) return res.status(400).json({ error: "Query required" });

    const langInstruction = language === "hindi" 
        ? "CRITICAL RULE: Answer in HINDI (Devanagari script)." 
        : "Answer in English.";

    const queryVector = await getEmbedding(query);

    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 6, 
      includeMetadata: true,
    });

    const matches = searchResult.matches || [];
    
  
    const context = matches
      .map((m, i) => `[Source: ${m.metadata?.source || "Legal Doc"}] Section ${m.metadata?.section}:\n${m.metadata?.text || ""}`)
      .join("\n\n");

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are LawSphere, a Universal Legal AI for India.
                
                ${langInstruction}

                STRICT INSTRUCTIONS:
                1. You have access to multiple Indian Laws (BNS, BNSS, BSA, IT Act, etc.).
                2. Answer ONLY using the provided context.
                3. **Explicitly mention the Act/Law Name** from the context (e.g. "According to BNS Section...").
                4. Format in Markdown.`
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

    res.json({
      formattedAnswer: answer,
      reasoning: "Universal Vector Search",
      semanticTags: matches.map(m => m.metadata?.source || "Law"), 
      retrievedSources: matches.map((m, i) => ({
        sourceNumber: i + 1,
        snippet: `[${m.metadata?.source}] ${m.metadata?.text?.substring(0, 150)}...`
      }))
    });

  } catch (error) {
    console.error("❌ Chat Error:", error);
    res.status(500).json({ formattedAnswer: "Server Error: " + error.message, retrievedSources: [] });
  }
});

router.post("/compare", async (req, res) => {
  try {
    const { section1, section2 } = req.body;
    
    const vec1 = await getEmbedding(section1);
    const vec2 = await getEmbedding(section2);

    const [result1, result2] = await Promise.all([
        index.namespace(NAMESPACE).query({ vector: vec1, topK: 3, includeMetadata: true }),
        index.namespace(NAMESPACE).query({ vector: vec2, topK: 3, includeMetadata: true })
    ]);

    const matches = [...(result1.matches || []), ...(result2.matches || [])];
    const uniqueContext = Array.from(new Set(matches.map(m => `[${m.metadata.source}] ${m.metadata.text}`))).join("\n\n---\n\n");

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are an expert Indian Legal AI. Compare the two requested topics using the provided context from various Indian Acts.`
            },
            {
                role: "user",
                content: `CONTEXT: ${uniqueContext}\nTASK: Compare "${section1}" and "${section2}" (Table Format).`
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
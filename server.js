require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pinecone } = require("@pinecone-database/pinecone");
const { pipeline } = require("@xenova/transformers");
const Groq = require("groq-sdk");

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;
const NAMESPACE = "default";
const LLM_MODEL = "llama-3.3-70b-versatile"; 

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX);

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});


let embedder = null;

async function loadModel() {
  console.log("🧠 Loading local embedding model (Xenova)...");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log("✅ Embedding Model loaded.");
}

async function getEmbedding(text) {
  if (!embedder) await loadModel();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data).map(Number);
}

async function optimizeQuery(userQuery) {
    try {
        const completion = await groq.chat.completions.create({
            messages:[
                {
                    role: "system",
                    content: `You are a Legal Search Optimizer. Convert the user's query into the Full Official Legal Act Name. Output ONLY the refined query.`
                },
                { role: "user", content: userQuery }
            ],
            model: LLM_MODEL,
            temperature: 0,
        });
        return completion.choices[0]?.message?.content?.trim() || userQuery;
    } catch (e) {
        return userQuery;
    }
}

const router = express.Router();
router.get("/", (req, res) => res.send("🚀 LawSphere Brain is Active"));

// 1. ASK ROUTE
router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`\n📩 Chat Query: "${query}"`);
    if (!query) return res.status(400).json({ error: "Query required" });

    const langInstruction = language === "hindi" 
        ? "CRITICAL RULE: Answer in HINDI (Devanagari script). Use simple legal Hindi." 
        : "Answer in English.";

    const queryVector = await getEmbedding(query);

    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 5,
      includeMetadata: true,
    });

    const matches = searchResult.matches || [];
    
    const context = matches
      .map((m, i) => `Source ${i + 1}:\n${m.metadata?.text || ""}`)
      .join("\n\n");

    const completion = await groq.chat.completions.create({
        messages:[
            {
                role: "system",
                content: `You are LawSphere, an expert legal AI for Bharatiya Nyaya Sanhita (BNS). 
                ${langInstruction}
                STRICT RULES:
                1. Answer ONLY using the provided context.
                2. Cite relevant Section numbers.
                3. Format in Markdown.`
            },
            { role: "user", content: `CONTEXT FOUND IN DATABASE:\n${context}\n\nUSER QUESTION:\n${query}` }
        ],
        model: LLM_MODEL,
        temperature: 0.1, 
    });

    res.json({
      formattedAnswer: answer,
      reasoning: "Vector Search",
      semanticTags: ["BNS", "Legal"],
      retrievedSources: matches.map((m, i) => ({
        sourceNumber: i + 1,
        snippet: `[${m.metadata?.source}] ${m.metadata?.text?.substring(0, 150)}...`
      }))
    });
  } catch (error) {
    res.status(500).json({ formattedAnswer: "Server Error", retrievedSources:[] });
  }
});

router.post("/compare", async (req, res) => {
  try {
    const { section1, section2 } = req.body;
    console.log(`⚖️ Comparing: ${section1} vs ${section2}`);

    if (!section1 || !section2) {
      return res.status(400).json({ error: "Both sections required" });
    }

    const vec1 = await getEmbedding(section1);
    const vec2 = await getEmbedding(section2);

    const [result1, result2] = await Promise.all([
        index.namespace(NAMESPACE).query({ vector: vec1, topK: 3, includeMetadata: true }),
        index.namespace(NAMESPACE).query({ vector: vec2, topK: 3, includeMetadata: true })
    ]);

    const matches = [...(result1.matches || []), ...(result2.matches || [])];
    const uniqueContext = Array.from(new Set(matches.map(m => m.metadata?.text))).join("\n\n---\n\n");

    if (!uniqueContext || uniqueContext.length < 50) {
        return res.json({ formattedAnswer: "I could not find these sections in the uploaded BNS PDF.", retrievedSources: [] });
    }

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are a strict legal expert for BNS India. Use ONLY context. Output Markdown Table.`
            },
            {
                role: "user",
                content: `CONTEXT: ${uniqueContext}\nTASK: Compare "${section1}" and "${section2}".`
            }
        ],
        model: LLM_MODEL,
        temperature: 0.1,
    });

    const answer = completion.choices[0]?.message?.content || "Comparison failed.";

    res.json({
      formattedAnswer: answer,
      reasoning: "RAG Comparison",
      semanticTags: ["Compare"],
      retrievedSources: []
    });

  } catch (error) {
    console.error("❌ Compare Error:", error);
    res.status(500).json({ formattedAnswer: "Error: " + error.message, retrievedSources: [] });
  }
});



app.use("/api", router);

app.listen(PORT, "0.0.0.0", async () => {
  await loadModel();
  console.log(`🚀 LawSphere Backend running on port ${PORT}`);
});
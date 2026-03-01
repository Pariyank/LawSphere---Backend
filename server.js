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
  console.log("✅ Embedding Model loaded.");
}

async function getEmbedding(text) {
  if (!embedder) await loadModel();

  const output = await embedder(text, { pooling: "mean", normalize: true });
  
  return Array.from(output.data).map(Number);
}

const router = express.Router();


router.get("/", (req, res) => {
    res.send("🚀 LawSphere Brain is Active");
});


router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`📩 Query: "${query}" | Language: ${language}`);

    if (!query) return res.status(400).json({ error: "Query required" });

    const langInstruction = language === "hindi" 
        ? "CRITICAL RULE: Answer the user's question entirely in HINDI (Devanagari script). Use simple, clear legal Hindi." 
        : "Answer in English.";

    const queryVector = await getEmbedding(query);

    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 15, 
      includeMetadata: true,
    });

    const matches = searchResult.matches || [];
    

    const context = matches
      .map((m, i) => `[Source Document: ${m.metadata?.source || "Legal Doc"}] \nContent: ${m.metadata?.text || ""}`)
      .join("\n\n----------------\n\n");

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are LawSphere, a Universal Legal AI Assistant for India.
                
                ${langInstruction}

                CRITICAL INSTRUCTIONS:
                1. You have access to a database of 50+ Indian Laws (Acts, Rules, Codes like BNS, BNSS, BSA, etc.).
                2. Answer ONLY using the provided 'Context'.
                3. **IDENTIFY THE SOURCE:** Look at the '[Source Document: ...]' tag. Explicitly mention which Act you are quoting (e.g. "According to the Minimum Wages Act...").
                4. If the user asks about a specific Act (e.g. "Act 11 of 1948"), scan the context for that specific detail.
                5. Do not hallucinate. If the answer is not in the text, say: "I could not find information regarding this specific query in the database."
                6. Format in Markdown (Bold key terms, Bullet points).`
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
      reasoning: "Universal Vector Search",
      semanticTags: matches.slice(0, 3).map(m => m.metadata?.source || "Law"), 
      retrievedSources: matches.slice(0, 5).map((m, i) => ({
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
    console.log(`⚖️ Comparing: ${section1} vs ${section2}`);

    if (!section1 || !section2) {
      return res.status(400).json({ error: "Both sections required" });
    }

    const vec1 = await getEmbedding(section1);
    const vec2 = await getEmbedding(section2);

    const [result1, result2] = await Promise.all([
        index.namespace(NAMESPACE).query({ vector: vec1, topK: 5, includeMetadata: true }),
        index.namespace(NAMESPACE).query({ vector: vec2, topK: 5, includeMetadata: true })
    ]);


    const matches = [...(result1.matches || []), ...(result2.matches || [])];
    const uniqueContext = Array.from(new Set(matches.map(m => `[Source: ${m.metadata?.source}] ${m.metadata?.text}`))).join("\n\n");

    if (!uniqueContext || uniqueContext.length < 50) {
        return res.json({ formattedAnswer: "I could not find relevant sections in the database to compare.", retrievedSources: [] });
    }

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are an expert Indian Legal AI.
                Compare the two requested topics using ONLY the provided Context.
                Mention which Act/Law each section belongs to.
                
                Output a clean **Markdown Table** comparing:
                - Definition
                - Punishment
                - Nature (Cognizable/Bailable)`
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
    console.log("✅ Comparison Sent.");

    res.json({
      formattedAnswer: answer,
      reasoning: "RAG Comparison",
      semanticTags: ["Comparison"],
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
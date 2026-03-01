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

router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`📩 Query: "${query}" | Language: ${language}`);

    if (!query) return res.status(400).json({ error: "Query required" });

    const langInstruction = language === "hindi" 
        ? "CRITICAL RULE: Answer in HINDI (Devanagari script)." 
        : "Answer in English.";

    const queryVector = await getEmbedding(query);

    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 15, 
      includeMetadata: true,
    });

    const matches = searchResult.matches || [];
    
    
    const context = matches
      .map((m, i) => `[Source Document: ${m.metadata?.source}] \nContent: ${m.metadata?.text}`)
      .join("\n\n----------------\n\n");


    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are LawSphere, a Universal Legal AI Assistant for India.
                
                ${langInstruction}

                CRITICAL INSTRUCTIONS:
                1. You have access to a database of 50+ Indian Laws (Acts, Rules, Codes).
                2. Your job is to find the answer to the user's question from the provided 'Context'.
                3. **IDENTIFY THE SOURCE:** Look at the '[Source Document: ...]' tag in the context. Tell the user which Act/Law you are quoting from.
                4. **DO NOT LIMIT TO BNS:** If the context comes from "Minimum Wages Act", quote that. If it comes from "IT Act", quote that.
                5. If the user asks about "Central Act 11 of 1948", look through the context to find which Act corresponds to that year/number.
                6. If the answer is NOT in the context, explicitly say: "I could not find information regarding this specific Act in my database."
                7. Format in clean Markdown.`
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
    
    const vec1 = await getEmbedding(section1);
    const vec2 = await getEmbedding(section2);

    const [result1, result2] = await Promise.all([
        index.namespace(NAMESPACE).query({ vector: vec1, topK: 5, includeMetadata: true }),
        index.namespace(NAMESPACE).query({ vector: vec2, topK: 5, includeMetadata: true })
    ]);

    const matches = [...(result1.matches || []), ...(result2.matches || [])];
    const uniqueContext = matches.map(m => `[Source: ${m.metadata.source}] ${m.metadata.text}`).join("\n\n");

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: `You are an expert Indian Legal AI. Compare the two requested topics using ONLY the provided Context.
                Mention which Act/Law the sections belong to.`
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

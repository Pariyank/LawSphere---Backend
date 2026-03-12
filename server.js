require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");
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
                    content: `You are a Legal Search Optimizer. Convert the user's query into the Full Official Legal Act Name.
                    Input: "Act No 4 of 1936" -> Output: "Payment of Wages Act, 1936"
                    Output ONLY the refined query.`
                },
                { role: "user", content: userQuery }
            ],
            model: LLM_MODEL,
            temperature: 0,
        });
        const refined = completion.choices[0]?.message?.content?.trim() || userQuery;
        return refined;
    } catch (e) {
        return userQuery;
    }
}

const router = express.Router();
router.get("/", (req, res) => res.send("🚀 LawSphere Brain is Active"));

// ---------------------------------------------------------
// 1. CHAT ROUTE
// ---------------------------------------------------------
router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`\n📩 Chat Query: "${query}"`);

    if (!query) return res.status(400).json({ error: "Query required" });

    const refinedQuery = await optimizeQuery(query);
    
    const searchString = refinedQuery + " whoever punished imprisonment fine explanation";
    const queryVector = await getEmbedding(searchString);

    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 15, 
      includeMetadata: true,
    });

    const matches = searchResult.matches ||[];
    const context = matches.map(m => `[ACT/DOCUMENT: ${m.metadata?.source}] \nCONTENT: ${m.metadata?.text}`).join("\n\n----------------\n\n");

    const langInstruction = language === "hindi" ? "Answer in HINDI (Devanagari script)." : "Answer in English.";

    const completion = await groq.chat.completions.create({
        messages:[
            {
                role: "system",
                content: `You are LawSphere, a strict Universal Legal Database Assistant for India.
                ${langInstruction}
                
                CRITICAL INSTRUCTIONS:
                1. **STRICT RAG MODE:** You must derive your answer EXCLUSIVELY from the 'CONTEXT FOUND IN DATABASE'.
                2. **NO OLD LAWS:** Do NOT reference the old Indian Penal Code (IPC) unless the context specifically provides it.
                3. **CHECK THE SOURCE:** Start your answer with: "According to the [Insert Document Name]..."
                4. **EXACT TEXT:** Do not make up punishments. If the text does not state the punishment, say "The exact punishment is not specified in the retrieved text."
                5. Format nicely in Markdown.`
            },
            { role: "user", content: `CONTEXT FOUND IN DATABASE:\n${context}\n\nUSER QUESTION:\n${query}` }
        ],
        model: LLM_MODEL,
        temperature: 0.1, 
    });

    res.json({
      formattedAnswer: completion.choices[0]?.message?.content || "No answer generated.",
      reasoning: "Universal Vector Search",
      semanticTags: matches.slice(0, 3).map(m => m.metadata?.source || "Law"),
      retrievedSources: matches.slice(0, 5).map((m, i) => ({
        sourceNumber: i + 1,
        snippet: `[${m.metadata?.source}] ${m.metadata?.text?.substring(0, 150)}...`
      }))
    });

  } catch (error) {
    console.error("❌ Chat Error:", error);
    res.status(500).json({ formattedAnswer: "Server Error: " + error.message, retrievedSources:[] });
  }
});

// ---------------------------------------------------------
// 2. COMPARE ROUTE
// ---------------------------------------------------------
router.post("/compare", async (req, res) => {
  try {
    const { section1, section2 } = req.body;
    const op1 = await optimizeQuery(section1);
    const op2 = await optimizeQuery(section2);

    const vec1 = await getEmbedding(op1 + " definition punishment");
    const vec2 = await getEmbedding(op2 + " definition punishment");

    const[result1, result2] = await Promise.all([
        index.namespace(NAMESPACE).query({ vector: vec1, topK: 6, includeMetadata: true }),
        index.namespace(NAMESPACE).query({ vector: vec2, topK: 6, includeMetadata: true })
    ]);

    const matches = [...(result1.matches || []), ...(result2.matches ||[])];
    const uniqueContext = Array.from(new Set(matches.map(m => `[Doc: ${m.metadata.source}] ${m.metadata.text}`))).join("\n\n");

    const completion = await groq.chat.completions.create({
        messages:[
            { role: "system", content: `Compare the two requested topics using ONLY the provided Context. Output a Markdown Table.` },
            { role: "user", content: `CONTEXT: ${uniqueContext}\nTASK: Compare "${section1}" and "${section2}".` }
        ],
        model: LLM_MODEL,
        temperature: 0.1,
    });

    res.json({
      formattedAnswer: completion.choices[0]?.message?.content || "Comparison failed.",
      semanticTags: ["Comparison"],
      retrievedSources:[]
    });
  } catch (error) {
    res.status(500).json({ formattedAnswer: "Error: " + error.message, retrievedSources:[] });
  }
});

// ---------------------------------------------------------
// 3. EXACT LOOKUP ROUTE (🟢 HYBRID SEARCH FIX)
// ---------------------------------------------------------
router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body; 
        console.log(`\n🔎 Exact Lookup -> Act: "${act}", Section: "${section}"`);

        // 1. Embed the general intent
        const searchString = `${act} Section ${section}`;
        const queryVector = await getEmbedding(searchString);

        // 2. Fetch a MASSIVE pool of chunks (Top 50) to guarantee the number is caught
        const searchResult = await index.namespace(NAMESPACE).query({
            vector: queryVector,
            topK: 50, 
            includeMetadata: true,
        });

        const matches = searchResult.matches ||[];

        // 3. Filter 1: Keep only chunks from the requested Act
        const cleanRequestedAct = act.replace(/[^a-zA-Z]/g, '').toLowerCase();
        let actMatches = matches.filter(m => {
            const cleanSource = (m.metadata.source || "").replace(/[^a-zA-Z]/g, '').toLowerCase();
            return cleanRequestedAct.includes(cleanSource) || cleanSource.includes(cleanRequestedAct);
        });

        // 4. Filter 2 (KEY FIX): Force Javascript to find the EXACT NUMBER in the text
        let finalMatches =[];
        if (section) {
            // Looks for the exact section number with word boundaries (so "10" doesn't match "102")
            const secRegex = new RegExp(`\\b${section}\\b`, 'i');
            finalMatches = actMatches.filter(m => secRegex.test(m.metadata.text) || secRegex.test(m.metadata.section));
        }

        // If we found the exact number, use those chunks. If not, fallback to the Act chunks.
        if (finalMatches.length > 0) {
            actMatches = finalMatches;
            console.log(`✅ Found ${finalMatches.length} chunks containing exact number "${section}"`);
        } else {
            console.log(`⚠️ Exact number "${section}" not found. Falling back to semantic matches.`);
        }

        // Take only the top 3 best matching chunks to avoid confusing the AI
        const context = actMatches.slice(0, 3).map(m => `[Doc: ${m.metadata.source}] ${m.metadata.text}`).join("\n\n");

        if (!context || context.length < 20) {
             return res.json({
                section: section, title: act, description: "This specific section could not be found in the database.", punishment: "N/A", cognizable: "N/A", bailable: "N/A", chapter: "N/A", cases:[]
            });
        }

        // 5. Strict Extraction Prompt
        const completion = await groq.chat.completions.create({
            messages:[
                {
                    role: "system",
                    content: `You are a Strict Legal Extraction Engine.
                    Your Task: Extract the EXACT details for Section ${section} from the context.
                    
                    CRITICAL RULES:
                    1. Do NOT hallucinate. If the details for Section ${section} are NOT in the context, output "N/A" or "Not specified".
                    2. DO NOT mix the title of one section with the number of another.
                    3. Output strictly Raw JSON. No markdown blocks.
                    
                    JSON FORMAT:
                    {
                        "section": "${section}",
                        "title": "Name of offence/rule",
                        "description": "EXACT STATEMENT OR CONTENT copied from context",
                        "punishment": "Exact penalty if mentioned, else 'N/A'",
                        "cognizable": "Yes/No/N/A",
                        "bailable": "Yes/No/N/A",
                        "chapter": "Chapter name/number",
                        "cases":[]
                    }`
                },
                {
                    role: "user",
                    content: `CONTEXT:\n${context}\n\nEXTRACT Section ${section}:`
                }
            ],
            model: LLM_MODEL,
            temperature: 0,
            response_format: { type: "json_object" } 
        });

        let rawOutput = completion.choices[0]?.message?.content || "{}";
        let cleanJson = rawOutput.replace(/```json/gi, "").replace(/```/g, "").trim();
        let parsedData = JSON.parse(cleanJson);

        const safeJson = {
            section: parsedData.section || parsedData.Section || section,
            title: parsedData.title || parsedData.Title || `Section ${section}`,
            description: parsedData.description || parsedData.Description || "Statement could not be extracted.",
            punishment: parsedData.punishment || parsedData.Punishment || "N/A",
            cognizable: parsedData.cognizable || parsedData.Cognizable || "N/A",
            bailable: parsedData.bailable || parsedData.Bailable || "N/A",
            chapter: parsedData.chapter || parsedData.Chapter || "N/A",
            cases: parsedData.cases || parsedData.Cases ||[]
        };

        res.json(safeJson);

    } catch (error) {
        console.error("❌ Lookup Error:", error);
        res.status(200).json({ 
            section: "Error", title: "Parsing Error", description: "Failed to extract exact text.", punishment: "N/A", cognizable: "N/A", bailable: "N/A", chapter: "N/A", cases:[]
        });
    }
});


app.use("/api", router);

app.listen(PORT, "0.0.0.0", async () => {
  await loadModel();
  console.log(`🚀 LawSphere Backend running on port ${PORT}`);
});
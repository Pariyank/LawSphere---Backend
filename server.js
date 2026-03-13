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

// 🟢 LEGAL QUERY OPTIMIZER (For Chat & Compare)
async function optimizeQuery(userQuery) {
    try {
        const completion = await groq.chat.completions.create({
            messages:[
                {
                    role: "system",
                    content: `You are a Legal Search Optimizer. Convert the user's query into the Full Official Legal Act Name.
                    Input: "Act No 4 of 1936" -> Output: "Payment of Wages Act, 1936"
                    Input: "Punishment for murder" -> Output: "Punishment for murder Bharatiya Nyaya Sanhita"
                    Output ONLY the refined query.`
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
                1. Answer EXCLUSIVELY from the 'CONTEXT FOUND IN DATABASE'.
                2. Do NOT reference the old Indian Penal Code (IPC) unless explicitly in the context.
                3. Start your answer with: "According to the [Insert Document Name]..."
                4. Format nicely in Markdown.`
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

    const matches =[...(result1.matches || []), ...(result2.matches ||[])];
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
// 3. EXACT LOOKUP ROUTE (🟢 IMPROVED FOR MISSING DATA)
// ---------------------------------------------------------
router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body; 
        console.log(`\n🔎 Exact Lookup -> Act: "${act}", Section: "${section}"`);

        // 1. Broad Search String
        const searchString = `${act} Section ${section} definition whoever commits punished imprisonment fine explanation`;
        const queryVector = await getEmbedding(searchString);

        // 2. Fetch a MASSIVE pool of chunks (Top 60)
        const searchResult = await index.namespace(NAMESPACE).query({
            vector: queryVector,
            topK: 60,
            includeMetadata: true,
        });

        let matches = searchResult.matches ||[];

        // 3. Attempt to Filter by Act Name
        const cleanRequestedAct = act.replace(/[^a-zA-Z]/g, '').toLowerCase();
        
        let actMatches = matches.filter(m => {
            const rawSource = m.metadata?.source || m.id || "";
            const cleanSource = rawSource.replace(/[^a-zA-Z]/g, '').toLowerCase();
            return cleanSource.includes(cleanRequestedAct) || cleanRequestedAct.includes(cleanSource);
        });

        // Failsafe
        if (actMatches.length === 0) {
            console.log(`⚠️ Act Name mismatch. Falling back to semantic matches.`);
            actMatches = matches; 
        } else {
            console.log(`📚 Found ${actMatches.length} chunks matching Act Name.`);
        }

        // 4. SMART SORTING: Push chunks containing the exact section number to the top
        if (section) {
            actMatches.sort((a, b) => {
                const textA = a.metadata?.text || "";
                const textB = b.metadata?.text || "";
                const hasA = textA.includes(section) ? 1 : 0;
                const hasB = textB.includes(section) ? 1 : 0;
                return hasB - hasA; 
            });
        }

        // 5. Take the top 8 best chunks
        const context = actMatches.slice(0, 8).map(m => `[Doc: ${m.metadata?.source}] ${m.metadata?.text}`).join("\n\n---\n\n");

        if (!context || context.length < 20) {
             return res.json({
                section: section, title: act, description: "This specific section could not be found in the database.", punishment: "N/A", cognizable: "N/A", bailable: "N/A", chapter: "N/A", cases:[]
            });
        }

        // 6. 🟢 UPDATED STRICT EXTRACTION PROMPT
        const completion = await groq.chat.completions.create({
            messages:[
                {
                    role: "system",
                    content: `You are a Legal Data Extraction Engine.
                    Your Task: Extract details of Section ${section} from the context.
                    
                    CRITICAL RULES:
                    1. "description": Extract the MAIN legal definition. If the context contains ANY text related to the section, use it. Do NOT just say "Data missing".
                    2. "punishment": Extract the exact penalty. If no specific punishment is listed in the context, write "Please refer to the Act for specific penalties".
                    3. If cognizable/bailable are not explicitly written, INFER them based on your general knowledge of Indian Law for this specific crime.
                    4. Output strictly Raw JSON. No markdown blocks.
                    
                    JSON FORMAT:
                    {
                        "section": "${section}",
                        "title": "Heading of the section",
                        "description": "EXACT STATEMENT OR CONTENT",
                        "punishment": "Exact penalty if mentioned, else 'Refer to Act'",
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
        
        console.log("✅ Extracted UI JSON:", parsedData.title);

        let cogVal = parsedData.cognizable || parsedData.Cognizable || "N/A";
        let bailVal = parsedData.bailable || parsedData.Bailable || "N/A";

        if (parsedData["Cognizable/Bailable"]) {
            const combined = parsedData["Cognizable/Bailable"].toLowerCase();
            cogVal = combined.includes("non-cognizable") ? "No" : (combined.includes("cognizable") ? "Yes" : "N/A");
            bailVal = combined.includes("non-bailable") ? "No" : (combined.includes("bailable") ? "Yes" : "N/A");
        }

        const safeJson = {
            section: parsedData.section || parsedData.Section || section,
            title: parsedData.title || parsedData.Title || `Section ${section}`,
            description: parsedData.description || parsedData.Description || "Statement could not be extracted.",
            punishment: parsedData.punishment || parsedData.Punishment || "N/A",
            cognizable: cogVal,
            bailable: bailVal,
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
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

// 🟢 UPGRADED CHAT OPTIMIZER (Smarter Act Routing)
async function optimizeQuery(userQuery) {
    try {
        const completion = await groq.chat.completions.create({
            messages:[
                {
                    role: "system",
                    content: `You are a Legal Search Optimizer for an Indian Law Vector Database.
                    Your job is to clarify the user's query so the database finds the right document.
                    
                    RULES:
                    1. If the query is about a general crime/offence (murder, theft, rape, cyber crime, defamation), append "Bharatiya Nyaya Sanhita".
                    2. If the query is about police procedure, FIR, arrest, or bail, append "Bharatiya Nagarik Suraksha Sanhita".
                    3. If the query is about evidence or witnesses, append "Bharatiya Sakshya Adhiniyam".
                    4. If the query mentions a specific topic like "wages", "marriage", "companies", DO NOT append BNS. Just clarify the topic (e.g., "Payment of Wages Act").
                    5. Output ONLY the optimized search string. No quotes, no conversational text.`
                },
                { role: "user", content: userQuery }
            ],
            model: LLM_MODEL,
            temperature: 0,
        });
        const refined = completion.choices[0]?.message?.content?.trim() || userQuery;
        console.log(`🔀 Chat Optimizer: "${userQuery}" -> "${refined}"`);
        return refined;
    } catch (e) {
        return userQuery;
    }
}

const router = express.Router();
router.get("/", (req, res) => res.send("🚀 LawSphere Brain is Active"));

// ---------------------------------------------------------
// 1. CHAT ROUTE (🟢 FIXED FOR ACCURATE RAG RETRIEVAL)
// ---------------------------------------------------------
router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`\n📩 Chat Query: "${query}"`);

    if (!query) return res.status(400).json({ error: "Query required" });

    // 1. Translate the query to target the right Act
    const refinedQuery = await optimizeQuery(query);
    
    // 🟢 FIX: REMOVED the "whoever punished imprisonment" anchor text. 
    // It was ruining the vector math for short queries like "punishment for murder".
    const queryVector = await getEmbedding(refinedQuery);

    // 2. Search Database (Increased TopK to ensure we find the exact section)
    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 25, // 🟢 Increased to 25 to give the AI plenty of context to read
      includeMetadata: true,
    });

    const matches = searchResult.matches ||[];
    
    console.log("🔎 Pinecone Chat Retrieval (Top 3):");
    matches.slice(0, 3).forEach(m => {
        console.log(`   -[${m.score.toFixed(2)}] Source: ${m.metadata?.source}`);
    });

    const context = matches.map(m => `[ACT/DOCUMENT: ${m.metadata?.source}] \nCONTENT: ${m.metadata?.text}`).join("\n\n----------------\n\n");

    const langInstruction = language === "hindi" ? "Answer in HINDI (Devanagari script)." : "Answer in English.";

    // 3. Generate Answer (🟢 Smarter formatting rules)
    const completion = await groq.chat.completions.create({
        messages:[
            {
                role: "system",
                content: `You are LawSphere, a brilliant and strict Legal AI Assistant for India.
                ${langInstruction}
                
                CRITICAL INSTRUCTIONS:
                1. **STRICT RAG MODE:** Answer EXCLUSIVELY using the 'CONTEXT FOUND IN DATABASE'. Do not use outside knowledge.
                2. **IDENTIFY THE LAW:** Read the [ACT/DOCUMENT: ...] tag in the context. Start your answer naturally by naming the Act (e.g., "Under the Bharatiya Nyaya Sanhita, 2023..." or "According to the Minimum Wages Act..."). DO NOT say "According to the PDF".
                3. **BE PRECISE:** If the user asks for a punishment, state the exact imprisonment term and fine as written in the context. Mention the Section number.
                4. **NO IPC:** Do NOT reference the old Indian Penal Code (IPC) or CrPC. India uses BNS and BNSS.
                5. **FALLBACK:** If the exact answer is genuinely missing from the context, reply: "I could not find the specific legal provision for this in the current database."`
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
// 2. COMPARE ROUTE (Preserved)
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
            { role: "system", content: `Compare the two requested topics using ONLY the provided Context. Output a Markdown Table. Do not mention IPC.` },
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
// 3. EXACT LOOKUP ROUTE (Preserved exactly as requested)
// ---------------------------------------------------------
router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body; 
        console.log(`\n🔎 Exact Lookup -> Act: "${act}", Section: "${section}"`);

        const searchString = `${act} Section ${section} definition whoever commits punished imprisonment fine explanation`;
        const queryVector = await getEmbedding(searchString);

        const searchResult = await index.namespace(NAMESPACE).query({
            vector: queryVector,
            topK: 60,
            includeMetadata: true,
        });

        let matches = searchResult.matches ||[];
        
        const cleanRequestedAct = act.replace(/[^a-zA-Z]/g, '').toLowerCase();
        
        let actMatches = matches.filter(m => {
            const rawSource = m.metadata?.source || m.id || "";
            const cleanSource = rawSource.replace(/[^a-zA-Z]/g, '').toLowerCase();
            return cleanSource.includes(cleanRequestedAct) || cleanRequestedAct.includes(cleanSource);
        });

        if (actMatches.length === 0) {
            console.log(`⚠️ Act Name mismatch. Falling back to semantic matches.`);
            actMatches = matches; 
        } else {
            console.log(`📚 Found ${actMatches.length} chunks matching Act Name.`);
        }

        if (section) {
            actMatches.sort((a, b) => {
                const textA = a.metadata?.text || "";
                const textB = b.metadata?.text || "";
                const hasA = textA.includes(section) ? 1 : 0;
                const hasB = textB.includes(section) ? 1 : 0;
                return hasB - hasA; 
            });
        }

        const context = actMatches.slice(0, 8).map(m => `[Doc: ${m.metadata?.source}] ${m.metadata?.text}`).join("\n\n---\n\n");

        if (!context || context.length < 20) {
             return res.json({
                section: section, title: act, description: "This specific section could not be found in the database.", punishment: "N/A", cognizable: "N/A", bailable: "N/A", chapter: "N/A", cases:[]
            });
        }

        const completion = await groq.chat.completions.create({
            messages:[
                {
                    role: "system",
                    content: `You are a Strict Legal Extraction Engine.
                    Your Task: Extract details of Section ${section} from the context.
                    
                    CRITICAL RULES:
                    1. "description": Extract the MAIN legal definition. If the context is just an index, say "Data missing from context".
                    2. "punishment": Extract the exact penalty. If it's a civil act with no punishment, write "N/A - Civil Section".
                    3. If cognizable/bailable are not explicitly written, INFER them based on your general knowledge of Indian Law.
                    4. Output strictly Raw JSON. No markdown blocks.
                    
                    JSON FORMAT:
                    {
                        "section": "${section}",
                        "title": "Heading of the section",
                        "description": "EXACT STATEMENT OR CONTENT",
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

// ---------------------------------------------------------
// 4. NEWS ROUTE
// ---------------------------------------------------------
router.get("/news", async (req, res) => {
    try {
        const apiKey = process.env.NEWS_API_KEY;
        const url = `https://gnews.io/api/v4/search?q=Supreme%20Court%20India%20OR%20Indian%20Laws&lang=en&country=in&max=10&apikey=${apiKey}`;
        const response = await axios.get(url);
        res.json(response.data.articles.map(a => ({ title: a.title, description: a.description, source: a.source.name, date: new Date(a.publishedAt).toDateString() })));
    } catch (error) {
        res.json([{ title: "News Unavailable", description: "Check internet/API.", source: "System", date: "Today" }]);
    }
});

app.use("/api", router);

app.listen(PORT, "0.0.0.0", async () => {
  await loadModel();
  console.log(`🚀 LawSphere Backend running on port ${PORT}`);
});
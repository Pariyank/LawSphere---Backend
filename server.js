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
        return completion.choices[0]?.message?.content?.trim() || userQuery;
    } catch (e) {
        return userQuery;
    }
}

const router = express.Router();
router.get("/", (req, res) => res.send("🚀 LawSphere Brain is Active"));

router.post("/ask", async (req, res) => {
  try {
    const { query, language } = req.body;
    console.log(`\n📩 Chat Query: "${query}"`);

    if (!query) return res.status(400).json({ error: "Query required" });

    const refinedQuery = await optimizeQuery(query);
    const queryVector = await getEmbedding(refinedQuery);

    const searchResult = await index.namespace(NAMESPACE).query({
      vector: queryVector,
      topK: 15, 
      includeMetadata: true,
    });

    const matches = searchResult.matches ||[];
    const context = matches.map(m => `[DOCUMENT: ${m.metadata?.source}] \nCONTENT: ${m.metadata?.text}`).join("\n\n----------------\n\n");

    const langInstruction = language === "hindi" ? "Answer in HINDI." : "Answer in English.";

    const completion = await groq.chat.completions.create({
        messages:[
            {
                role: "system",
                content: `You are LawSphere, a strict Universal Legal Database Assistant for India.
                ${langInstruction}
                CRITICAL INSTRUCTIONS:
                1. Answer ONLY using the 'CONTEXT FOUND IN DATABASE'.
                2. NEVER mention the old IPC or CrPC unless explicitly in the text.
                3. Start your answer with: "According to the[Insert Document Name]..."
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
    res.status(500).json({ formattedAnswer: "Server Error", retrievedSources:[] });
  }
});


router.post("/compare", async (req, res) => {
    try {
      const { section1, section2 } = req.body;
      const vec1 = await getEmbedding(section1);
      const vec2 = await getEmbedding(section2);
  
      const[result1, result2] = await Promise.all([
          index.namespace(NAMESPACE).query({ vector: vec1, topK: 5, includeMetadata: true }),
          index.namespace(NAMESPACE).query({ vector: vec2, topK: 5, includeMetadata: true })
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


router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body; 
        console.log(`\n🔎 Exact Lookup -> Act: "${act}", Section: "${section}"`);

        const searchString = `Document: ${act} Section ${section} definition punishment statement`;
        const queryVector = await getEmbedding(searchString);

        const searchResult = await index.namespace(NAMESPACE).query({
            vector: queryVector,
            topK: 15,
            includeMetadata: true,
        });

        const matches = searchResult.matches ||[];
        
  
        const cleanRequestedAct = act.replace(/[^a-zA-Z]/g, '').toLowerCase();
        
        const relevantMatches = matches.filter(m => {
            const cleanSource = (m.metadata.source || "").replace(/[^a-zA-Z]/g, '').toLowerCase();
            return cleanRequestedAct.includes(cleanSource) || cleanSource.includes(cleanRequestedAct);
        });

        const context = relevantMatches.map(m => m.metadata.text).join("\n\n");

        if (!context || context.length < 20) {
             return res.json({
                section: section, 
                title: act, 
                description: "This specific section could not be found in the provided Act.", 
                punishment: "N/A", cognizable: "N/A", bailable: "N/A", chapter: "N/A", cases:[]
            });
        }
        const completion = await groq.chat.completions.create({
            messages:[
                {
                    role: "system",
                    content: `You are a Strict Legal Extraction Engine.
                    Your Task: Extract the EXACT text and details of Section ${section} from the provided context.
                    
                    CRITICAL RULES:
                    1. Do NOT hallucinate. If Section ${section} is NOT in the context, write "Not Found in Context".
                    2. "description": This must be the EXACT STATEMENT/CONTENT of the section copied from the context. Do not paraphrase.
                    3. If punishment, bailable, or cognizable details are missing, output "N/A".
                    4. Output strictly Raw JSON. No markdown blocks.
                    
                    JSON FORMAT:
                    {
                        "section": "${section}",
                        "title": "Heading of the section",
                        "description": "EXACT STATEMENT OR CONTENT of the section",
                        "punishment": "Exact penalty if mentioned, else 'N/A'",
                        "cognizable": "Yes/No/N/A",
                        "bailable": "Yes/No/N/A",
                        "chapter": "Chapter name/number",
                        "cases":[]
                    }`
                },
                {
                    role: "user",
                    content: `CONTEXT (From ${act}):\n${context}\n\nEXTRACT Section ${section}:`
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
            description: parsedData.description || parsedData.Description || "Exact statement could not be extracted.",
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
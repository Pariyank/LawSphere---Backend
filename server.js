require("dotenv").config();
const express = require("express");
const cors = require("cors");
const admin = require("firebase-admin");
const { Pinecone } = require("@pinecone-database/pinecone");
const { pipeline } = require("@xenova/transformers");
const Groq = require("groq-sdk");

const app = express();
app.use(cors());
app.use(express.json());

// ================= FIREBASE ADMIN SETUP =================
let serviceAccount;

if (process.env.FIREBASE_SERVICE_ACCOUNT) {
    serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
} else {
    serviceAccount = require("./firebase-service-account.json");
}

if (!admin.apps.length) {
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount)
    });
}
const db = admin.firestore();

// ================= CONFIG & SERVICES =================
const PORT = process.env.PORT || 3000;
const NAMESPACE = "default";
const LLM_MODEL = "llama-3.3-70b-versatile"; 

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX);
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

let embedder = null;
async function loadModel() {
  console.log("🧠 Loading local embedding model...");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log("✅ Model loaded.");
}

async function getEmbedding(text) {
  if (!embedder) await loadModel();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data).map(Number);
}

// 🟢 STEP 1: LAYMAN TO LEGAL OPTIMIZER
async function optimizeQuery(userQuery) {
    try {
        const completion = await groq.chat.completions.create({
            messages: [{
                role: "system",
                content: `You are an Indian Legal Search Optimizer. Translate layman terms to formal legal terminology used in Bare Acts. 
                Identify the relevant Act (BNS, BNSS, IT Act, etc.). Return ONLY the optimized query keywords.`
            }, { role: "user", content: userQuery }],
            model: LLM_MODEL, temperature: 0
        });
        return completion.choices[0]?.message?.content?.trim() || userQuery;
    } catch (e) { return userQuery; }
}

const router = express.Router();
router.get("/", (req, res) => res.send("🚀 LawSphere Hybrid Engine Online"));

// ---------------------------------------------------------
// 🟢 CHAT ROUTE (RAG with Firestore Grounding)
// ---------------------------------------------------------
router.post("/ask", async (req, res) => {
    try {
        const { query, language } = req.body;
        console.log(`📩 Chat: ${query}`);

        const refinedQuery = await optimizeQuery(query);
        const queryVector = await getEmbedding(refinedQuery);

        const searchResult = await index.namespace(NAMESPACE).query({
            vector: queryVector, topK: 8, includeMetadata: true
        });

        const matches = searchResult.matches || [];
        let contextText = "";
        let sources = [];

        for (const m of matches) {
            const docId = m.metadata.firestore_id;
            const doc = await db.collection("legal_sections").doc(docId).get();
            if (doc.exists) {
                const d = doc.data();
                contextText += `[LAW: ${d.act_name} | SEC: ${d.section_raw}]\nTEXT: ${d.content}\n\n`;
                sources.push(`[${d.act_name}] ${d.section_raw}`);
            }
        }

        const langRule = language === "hindi" ? "Answer in HINDI (Devanagari)." : "Answer in English.";
        const completion = await groq.chat.completions.create({
            messages: [{
                role: "system",
                content: `You are LawSphere AI. ${langRule}
                1. Answer ONLY using the provided Context. 
                2. Explain in simple, non-legalese language.
                3. Cite Act and Section names.
                4. If not in context, say 'Information not present in database.'`
            }, { role: "user", content: `CONTEXT:\n${contextText}\n\nUSER QUESTION: ${query}` }],
            model: LLM_MODEL, temperature: 0.1
        });

        res.json({
            formattedAnswer: completion.choices[0].message.content,
            retrievedSources: sources.slice(0, 5).map((s, i) => ({ sourceNumber: i + 1, snippet: s }))
        });
    } catch (error) {
        res.status(500).json({ formattedAnswer: "Connection error with AI brain." });
    }
});

// ---------------------------------------------------------
// 🟢 LIBRARY LOOKUP (Direct Firestore - 100% Accurate)
// ---------------------------------------------------------
router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body;
        const cleanSec = section.replace(/[^0-9A-Z]/ig, '');
        
        const snapshot = await db.collection("legal_sections")
            .where("act_name", "==", act)
            .where("section_number", "==", cleanSec)
            .limit(1).get();

        if (snapshot.empty) {
            return res.json({ section: "N/A", title: "Not Found", description: "This section was not found in our database.", punishment: "N/A" });
        }

        const data = snapshot.docs[0].data();
        
        // Use AI only to extract the tags for the UI Card
        const completion = await groq.chat.completions.create({
            messages: [{
                role: "system",
                content: `Extract from text and return JSON only: {"punishment":"...", "cognizable":"Yes/No/NA", "bailable":"Yes/No/NA"}`
            }, { role: "user", content: data.content }],
            model: LLM_MODEL, temperature: 0, response_format: { type: "json_object" }
        });

        const tags = JSON.parse(completion.choices[0].message.content);

        res.json({
            section: data.section_raw,
            title: data.title,
            description: data.content, // 100% Original Text
            punishment: tags.punishment || "N/A",
            cognizable: tags.cognizable || "N/A",
            bailable: tags.bailable || "N/A",
            chapter: data.chapter_name || "General"
        });
    } catch (e) { res.status(500).send("Error"); }
});

// ---------------------------------------------------------
// 🟢 COMPARE ROUTE
// ---------------------------------------------------------
router.post("/compare", async (req, res) => {
    try {
        const { section1, section2 } = req.body;
        const v1 = await getEmbedding(section1), v2 = await getEmbedding(section2);
        const [r1, r2] = await Promise.all([
            index.query({ vector: v1, topK: 3, includeMetadata: true }),
            index.query({ vector: v2, topK: 3, includeMetadata: true })
        ]);

        const context = [...r1.matches, ...r2.matches].map(m => m.metadata.text).join("\n\n");
        const completion = await groq.chat.completions.create({
            messages: [{ role: "system", content: "Compare using ONLY context. Output Markdown Table." }, { role: "user", content: `CONTEXT:\n${context}\n\nCOMPARE: ${section1} vs ${section2}` }],
            model: LLM_MODEL, temperature: 0.1
        });
        res.json({ formattedAnswer: completion.choices[0].message.content });
    } catch (e) { res.json({ formattedAnswer: "Comparison failed." }); }
});

app.use("/api", router);
app.listen(PORT, "0.0.0.0", async () => { await loadModel(); console.log(`🚀 Server on port ${PORT}`); });
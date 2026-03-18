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

// ================= 1. FIREBASE ADMIN SETUP (RENDER COMPATIBLE) =================
let serviceAccount;
try {
    if (process.env.FIREBASE_SERVICE_ACCOUNT) {
        // Reads from Render Environment Variable
        serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
    } else {
        // Reads from local file
        serviceAccount = require("./firebase-service-account.json");
    }

    if (!admin.apps.length) {
        admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    }
} catch (e) {
    console.error("❌ Firebase Initialization Error:", e.message);
}
const db = admin.firestore();

// ================= 2. CONFIG & SERVICES =================
const PORT = process.env.PORT || 3000;
const NAMESPACE = "default";
const LLM_MODEL = "llama-3.3-70b-versatile"; 

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX || "lawsphere-index");
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ================= 3. LOCAL EMBEDDING ENGINE =================
let embedder = null;
async function loadModel() {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
}

async function getEmbedding(text) {
    if (!embedder) await loadModel();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data).map(Number);
}

// 🟢 SMART QUERY OPTIMIZER
async function optimizeQuery(userQuery) {
    try {
        const completion = await groq.chat.completions.create({
            messages: [{
                role: "system",
                content: "You are a Legal Search Optimizer. Convert query to Official Act Name keywords. Example: 'Act 4 of 1936' -> 'Payment of Wages Act 1936'. Return ONLY keywords."
            }, { role: "user", content: userQuery }],
            model: LLM_MODEL, temperature: 0
        });
        return completion.choices[0]?.message?.content?.trim() || userQuery;
    } catch (e) { return userQuery; }
}

const router = express.Router();
router.get("/", (req, res) => res.send("🚀 LawSphere Engine Active"));

// ---------------------------------------------------------
// 🟢 ROUTE 1: CHATBOT (/api/ask)
// ---------------------------------------------------------
router.post("/ask", async (req, res) => {
    try {
        const { query, language } = req.body;
        const refinedQuery = await optimizeQuery(query);
        const queryVector = await getEmbedding(refinedQuery);

        const result = await index.namespace(NAMESPACE).query({ vector: queryVector, topK: 10, includeMetadata: true });
        
        let contextText = "";
        let sources = [];

        for (const match of result.matches) {
            const doc = await db.collection("legal_sections").doc(match.metadata.firestore_id).get();
            if (doc.exists) {
                const d = doc.data();
                contextText += `[ACT: ${d.act_name} | SEC: ${d.section_raw}]\nTEXT: ${d.content}\n\n`;
                sources.push({ sourceNumber: sources.length + 1, snippet: `[${d.act_name}] ${d.section_raw}` });
            }
        }

        const lang = language === "hindi" ? "Answer in HINDI." : "Answer in English.";
        const completion = await groq.chat.completions.create({
            messages: [{
                role: "system",
                content: `You are LawSphere AI. ${lang} Answer ONLY using provided Context. Use simple words. If not found, say 'This is not present in the database.'`
            }, { role: "user", content: `CONTEXT:\n${contextText}\n\nQUESTION: ${query}` }],
            model: LLM_MODEL, temperature: 0.1
        });

        res.json({ formattedAnswer: completion.choices[0].message.content, retrievedSources: sources.slice(0, 5) });
    } catch (error) { res.status(500).json({ formattedAnswer: "Brain connection error." }); }
});

// ---------------------------------------------------------
// 🟢 ROUTE 2: LIBRARY LOOKUP (/api/lookup) - DETERMINISTIC
// ---------------------------------------------------------
router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body;
        console.log(`🔎 Lookup -> Act: ${act}, Sec: ${section}`);

        // 🟢 FIX: Clean the search section to match numeric format
        const cleanSearchSec = section.replace(/[^0-9A-Z]/ig, '');

        // 1. Try exact match first
        let snapshot = await db.collection("legal_sections")
            .where("act_name", "==", act)
            .where("section_number", "==", cleanSearchSec)
            .limit(1).get();

        // 2. If fails, try matching the start of act name (Fuzzy match for commas/years)
        if (snapshot.empty) {
            console.log("   ➤ No exact match, trying fuzzy act match...");
            snapshot = await db.collection("legal_sections")
                .where("section_number", "==", cleanSearchSec)
                .limit(20).get();
        }

        // Filter the results in JS to ensure the correct Act is chosen
        const doc = snapshot.docs.find(d => {
            const dbAct = d.data().act_name.replace(/[^a-zA-Z]/g, '').toLowerCase();
            const reqAct = act.replace(/[^a-zA-Z]/g, '').toLowerCase();
            return dbAct.includes(reqAct) || reqAct.includes(dbAct);
        });

        if (!doc) {
            return res.json({ section: "N/A", title: "Not Found", description: "This specific section was not found in the selected Act.", punishment: "N/A" });
        }

        const data = doc.data();

        // Use AI to extract UI tags
        const completion = await groq.chat.completions.create({
            messages: [{
                role: "system",
                content: 'Return JSON only: {"punishment":"...", "cognizable":"Yes/No/NA", "bailable":"Yes/No/NA"}. Infer from text.'
            }, { role: "user", content: data.content }],
            model: LLM_MODEL, temperature: 0, response_format: { type: "json_object" }
        });

        const tags = JSON.parse(completion.choices[0].message.content);

        res.json({
            section: data.section_raw,
            title: data.title,
            description: data.content,
            punishment: tags.punishment || "N/A",
            cognizable: tags.cognizable || "N/A",
            bailable: tags.bailable || "N/A",
            chapter: data.chapter_name || "General"
        });
    } catch (e) { res.status(500).json({ description: "Lookup Error" }); }
});

// ---------------------------------------------------------
// 🟢 ROUTE 3: COMPARE & NEWS
// ---------------------------------------------------------
router.post("/compare", async (req, res) => {
    try {
        const { section1, section2 } = req.body;
        const v1 = await getEmbedding(section1), v2 = await getEmbedding(section2);
        const [r1, r2] = await Promise.all([
            index.namespace(NAMESPACE).query({ vector: v1, topK: 3, includeMetadata: true }),
            index.namespace(NAMESPACE).query({ vector: v2, topK: 3, includeMetadata: true })
        ]);
        const context = [...r1.matches, ...r2.matches].map(m => m.metadata.text).join("\n\n");
        const completion = await groq.chat.completions.create({
            messages: [{ role: "system", content: "Compare these two laws. Output Markdown Table." }, { role: "user", content: `CONTEXT:\n${context}\n\nCOMPARE: ${section1} vs ${section2}` }],
            model: LLM_MODEL, temperature: 0.1
        });
        res.json({ formattedAnswer: completion.choices[0].message.content });
    } catch (e) { res.json({ formattedAnswer: "Comparison failed." }); }
});



app.use("/api", router);
app.listen(PORT, "0.0.0.0", async () => { await loadModel(); console.log(`🚀 LawSphere Active on Port ${PORT}`); });
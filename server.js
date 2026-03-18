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

// ================= 1. FIREBASE ADMIN SETUP (RENDER FIX) =================
try {
    let serviceAccount;
    if (process.env.FIREBASE_SERVICE_ACCOUNT) {
        // 🟢 FIX: Correctly handle the private key string for Render environment variables
        serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
        if (typeof serviceAccount.private_key === 'string') {
            serviceAccount.private_key = serviceAccount.private_key.replace(/\\n/g, '\n');
        }
    } else {
        serviceAccount = require("./firebase-service-account.json");
    }

    if (!admin.apps.length) {
        admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    }
    console.log("✅ Firebase Admin Initialized Successfully");
} catch (e) {
    console.error("❌ Firebase Init Error:", e.message);
}
const db = admin.firestore();

// ================= 2. SERVICES =================
const PORT = process.env.PORT || 3000;
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX || "lawsphere-index");
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ================= 3. EMBEDDING ENGINE (OPTIMIZED) =================
let embedder = null;
async function loadModel() {
    try {
        console.log("🧠 Loading embedding model... (This might take a moment on Render)");
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
        console.log("✅ Model loaded.");
    } catch (err) {
        console.error("❌ Model Load Error (Check Render RAM limits):", err.message);
    }
}

async function getEmbedding(text) {
    if (!embedder) await loadModel();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data).map(Number);
}

const normalize = (str) => String(str || "").replace(/[^a-zA-Z0-9]/g, '').toLowerCase();

// ================= 4. ROUTES =================
const router = express.Router();

router.get("/", (req, res) => res.send("🚀 LawSphere Engine is Alive"));

router.post("/ask", async (req, res) => {
    try {
        const { query, language } = req.body;
        const queryVector = await getEmbedding(query);
        const result = await index.namespace("default").query({ vector: queryVector, topK: 10, includeMetadata: true });
        
        let contextText = "";
        let sourceList = [];
        for (const match of result.matches) {
            const fId = match.metadata?.firestore_id;
            if (fId) {
                const doc = await db.collection("legal_sections").doc(fId).get();
                if (doc.exists) {
                    const d = doc.data();
                    contextText += `[LAW: ${d.act_name} | SEC: ${d.section_raw}]\nTEXT: ${d.content}\n\n`;
                    sourceList.push({ sourceNumber: sourceList.length + 1, snippet: `[${d.act_name}] ${d.section_raw}` });
                }
            }
        }

        const completion = await groq.chat.completions.create({
            messages: [{ role: "system", content: "You are LawSphere AI. Answer using ONLY provided context. Explain simply. No old laws (IPC/CrPC)." }, 
                       { role: "user", content: `CONTEXT:\n${contextText}\n\nQUESTION: ${query}` }],
            model: "llama-3.3-70b-versatile", temperature: 0.1
        });

        res.json({ formattedAnswer: completion.choices[0].message.content, retrievedSources: sourceList.slice(0, 5) });
    } catch (error) { 
        console.error("Ask Error:", error.message);
        res.status(500).json({ error: error.message }); 
    }
});

router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body;
        const snapshot = await db.collection("legal_sections").where("act_name", "==", act).get();
        const searchNorm = normalize(section);
        const doc = snapshot.docs.find(d => normalize(d.data().section_number) === searchNorm || normalize(d.data().section_raw) === searchNorm);

        if (!doc) return res.json({ title: "Not Found", description: "Section not found." });
        const data = doc.data();
        const completion = await groq.chat.completions.create({
            messages: [{ role: "system", content: 'Return JSON: {"punishment":"...", "cognizable":"Yes/No/NA", "bailable":"Yes/No/NA"}' }, { role: "user", content: data.content }],
            model: "llama-3.3-70b-versatile", temperature: 0, response_format: { type: "json_object" }
        });
        const tags = JSON.parse(completion.choices[0].message.content);
        res.json({ section: data.section_raw, title: data.title, description: data.content, punishment: tags.punishment, cognizable: tags.cognizable, bailable: tags.bailable, chapter: data.chapter_name });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

router.post("/compare", async (req, res) => {
    try {
        const { act1, sec1, act2, sec2 } = req.body;
        const [snap1, snap2] = await Promise.all([
            db.collection("legal_sections").where("act_name", "==", act1).get(),
            db.collection("legal_sections").where("act_name", "==", act2).get()
        ]);
        const findMatch = (snap, s) => snap.docs.find(d => normalize(d.data().section_number) === normalize(s) || normalize(d.data().section_raw) === normalize(s));
        const d1 = findMatch(snap1, sec1), d2 = findMatch(snap2, sec2);
        if (!d1 || !d2) return res.json({ formattedAnswer: "One or both sections not found." });
        const completion = await groq.chat.completions.create({
            messages: [{ role: "system", content: "Compare these laws in a Markdown table." }, { role: "user", content: `1: ${d1.data().content}\n2: ${d2.data().content}` }],
            model: "llama-3.3-70b-versatile", temperature: 0.1
        });
        res.json({ formattedAnswer: completion.choices[0].message.content });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

app.use("/api", router);
app.listen(PORT, "0.0.0.0", async () => { await loadModel(); console.log(`🚀 Port ${PORT}`); });
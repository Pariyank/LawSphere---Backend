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

// ================= 1. FIREBASE ADMIN SETUP =================
try {
    let serviceAccount;
    if (process.env.FIREBASE_SERVICE_ACCOUNT) {
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
} catch (e) {
    console.error("❌ Firebase Init Error:", e.message);
}
const db = admin.firestore();

// ================= 2. SERVICES =================
const PORT = process.env.PORT || 3000;
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX || "lawsphere-index");
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ================= 3. OPTIMIZED EMBEDDING ENGINE =================
let embedder = null;

async function loadModel() {
    if (embedder) return; // Don't reload if already loaded
    try {
        console.log("🧠 Loading local embedding model...");
        // 🟢 MEMORY FIX: Force CPU only and lightweight configuration
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            revision: "main",
            quantized: true // Uses much less RAM
        });
        console.log("✅ Model loaded.");
    } catch (err) {
        console.error("❌ Model Load Error:", err.message);
        throw new Error("AI Model failed to load due to server memory limits.");
    }
}

async function getEmbedding(text) {
    await loadModel();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data).map(Number);
}

const normalize = (s) => String(s || "").replace(/[^a-zA-Z0-9]/g, '').toLowerCase();

// ================= 4. ROUTES =================
const router = express.Router();

router.get("/", (req, res) => res.send("🚀 LawSphere Engine Active"));

// 🟢 CHAT ROUTE with Enhanced Error Catching
router.post("/ask", async (req, res) => {
    try {
        const { query, language } = req.body;
        console.log(`📩 Chat Query: ${query}`);

        // 1. Get Embedding
        const queryVector = await getEmbedding(query);

        // 2. Search Pinecone
        const result = await index.namespace("default").query({ 
            vector: queryVector, 
            topK: 8, 
            includeMetadata: true 
        });
        
        if (!result.matches || result.matches.length === 0) {
            return res.json({ formattedAnswer: "I couldn't find any relevant laws for this in the database.", retrievedSources: [] });
        }

        // 3. Fetch Context from Firestore
        let contextText = "";
        let sourceList = [];
        
        // Concurrent fetch to save time
        const docPromises = result.matches.map(match => {
            const fId = match.metadata?.firestore_id;
            return fId ? db.collection("legal_sections").doc(fId).get() : null;
        });

        const docs = await Promise.all(docPromises);

        docs.forEach((doc, i) => {
            if (doc && doc.exists) {
                const d = doc.data();
                contextText += `[ACT: ${d.act_name} | SEC: ${d.section_raw}]\nTEXT: ${d.content}\n\n`;
                sourceList.push({ sourceNumber: sourceList.length + 1, snippet: `[${d.act_name}] ${d.section_raw}` });
            }
        });

        // 4. Groq Inference
        const lang = language === "hindi" ? "Answer in HINDI (Devanagari)." : "Answer in English.";
        const completion = await groq.chat.completions.create({
            messages: [
                { role: "system", content: `You are LawSphere AI. ${lang} Answer ONLY using provided context. Keep it simple.` }, 
                { role: "user", content: `CONTEXT:\n${contextText}\n\nQUESTION: ${query}` }
            ],
            model: "llama-3.3-70b-versatile", 
            temperature: 0.1
        });

        res.json({ 
            formattedAnswer: completion.choices[0].message.content, 
            retrievedSources: sourceList.slice(0, 5) 
        });

    } catch (error) { 
        console.error("❌ ASK ERROR:", error.message);
        // 🟢 DEBUG: Send the actual error message to Android instead of just 500
        res.status(500).json({ 
            formattedAnswer: `Server Error: ${error.message}. Please check Render logs.`, 
            retrievedSources: [] 
        }); 
    }
});

// 🟢 LOOKUP ROUTE (The one that works)
router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body;
        const snapshot = await db.collection("legal_sections").where("act_name", "==", act).get();
        const searchNorm = normalize(section);
        const doc = snapshot.docs.find(d => normalize(d.data().section_number) === searchNorm || normalize(d.data().section_raw) === searchNorm);

        if (!doc) return res.json({ title: "Not Found", description: "Section not found in this Act." });
        
        const data = doc.data();
        const completion = await groq.chat.completions.create({
            messages: [{ role: "system", content: 'Return JSON: {"punishment":"...", "cognizable":"Yes/No/NA", "bailable":"Yes/No/NA"}' }, { role: "user", content: data.content }],
            model: "llama-3.3-70b-versatile", 
            temperature: 0, 
            response_format: { type: "json_object" }
        });
        const tags = JSON.parse(completion.choices[0].message.content);
        res.json({ section: data.section_raw, title: data.title, description: data.content, punishment: tags.punishment, cognizable: tags.cognizable, bailable: tags.bailable, chapter: data.chapter_name });
    } catch (e) { 
        res.status(500).json({ error: e.message }); 
    }
});

app.use("/api", router);

app.listen(PORT, "0.0.0.0", async () => { 
    // We don't pre-load model here to save startup memory. 
    // It will load on the first chat request.
    console.log(`🚀 Server listening on port ${PORT}`); 
});
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
} catch (e) { console.error("Firebase Init Error"); }
const db = admin.firestore();

const PORT = process.env.PORT || 3000;
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX || "lawsphere-index");
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

let embedder = null;
async function loadModel() {
    if (embedder) return;
    try {
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            revision: "main",
            quantized: true 
        });
    } catch (err) { console.error("Model Load Error"); }
}

async function getEmbedding(text) {
    await loadModel();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data).map(Number);
}

const normalize = (s) => String(s || "").replace(/[^a-zA-Z0-9]/g, '').toLowerCase();


const router = express.Router();

router.get("/", (req, res) => res.send("🚀 LawSphere Engine Stable & Formatted"));

router.post("/ask", async (req, res) => {
    try {
        const { query, language } = req.body;
        const queryVector = await getEmbedding(query);

        const result = await index.namespace("default").query({ 
            vector: queryVector, 
            topK: 10, 
            includeMetadata: true 
        });
        
        let contextText = "";
        let sourceList = [];
        
        const docPromises = result.matches.map(match => {
            const fId = match.metadata?.firestore_id;
            return fId ? db.collection("legal_sections").doc(fId).get() : null;
        });

        const docs = await Promise.all(docPromises);
        docs.forEach((doc) => {
            if (doc && doc.exists) {
                const d = doc.data();
                contextText += `[ACT: ${d.act_name} | SEC: ${d.section_raw}]\nHEADING: ${d.title}\nTEXT: ${d.content}\n\n`;
                sourceList.push({ sourceNumber: sourceList.length + 1, snippet: `[${d.act_name}] ${d.section_raw}` });
            }
        });

        const lang = language === "hindi" ? "HINDI (Devanagari)" : "English";

        const completion = await groq.chat.completions.create({
            messages: [
                { 
                    role: "system", 
                    content: `You are LawSphere AI, a professional Legal Assistant. 
                    Answer in ${lang} using ONLY the provided context. 
                    
                    FORMATTING RULES:
                    1. Use # for the main Act Name.
                    2. Use > for the exact statutory provision.
                    3. Use ### for "Simple Explanation" and "Punishment".
                    4. Use bullet points for clarity.
                    5. If not in context, say 'This information is not present in the database.'` 
                }, 
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
        res.status(500).json({ formattedAnswer: "Server Error: " + error.message }); 
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
            model: "llama-3.3-70b-versatile", 
            temperature: 0, 
            response_format: { type: "json_object" }
        });
        const tags = JSON.parse(completion.choices[0].message.content);
        res.json({ section: data.section_raw, title: data.title, description: data.content, punishment: tags.punishment, cognizable: tags.cognizable, bailable: tags.bailable, chapter: data.chapter_name });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

router.get("/list-acts", (req, res) => {
    try {
        const actListData = fs.readFileSync("./legal_acts.json", "utf8");
        res.json(JSON.parse(actListData));
    } catch (e) {
        res.status(500).json({ error: "Could not load act list" });
    }
});

router.get("/offline-data", (req, res) => {
    try {
        const offlineData = fs.readFileSync("./offline_critical.json", "utf8");
        res.json(JSON.parse(offlineData));
    } catch (e) {
        res.status(500).json({ error: "Could not load offline data" });
    }
});

router.get("/bnss-forms", (req, res) => {
    try {
        const formsData = fs.readFileSync("./bnss_forms.json", "utf8");
        res.json(JSON.parse(formsData));
    } catch (e) {
        console.error("Error loading forms:", e);
        res.status(500).json({ error: "Could not load BNSS forms list" });
    }
});

app.use("/api", router);
app.listen(PORT, "0.0.0.0", () => console.log(`🚀 Stable & Formatted Server on ${PORT}`));
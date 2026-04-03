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

const normalize = (str) => {
    return String(str || "")
        .replace(/section/gi, '')
        .replace(/article/gi, '')
        .replace(/[^a-zA-Z0-9]/g, '')
        .toLowerCase()
        .trim();
};


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
        console.log(`🔎 Searching Firestore: Act [${act}] Section [${section}]`);

        const snapshot = await db.collection("legal_sections")
            .where("act_name", "==", act)
            .get();

        if (snapshot.empty) {
            return res.json({ title: "Act Not Found", description: "This Act is not loaded in the database." });
        }

        const userInputNorm = normalize(section);
        
        const doc = snapshot.docs.find(d => {
            const data = d.data();
            const dbSecNumNorm = normalize(data.section_number);
            const dbSecRawNorm = normalize(data.section_raw);
            
            return dbSecNumNorm === userInputNorm || dbSecRawNorm === userInputNorm;
        });

        if (!doc) {
            return res.json({ section: section, title: "Not Found", description: `Could not find '${section}' in ${act}.`, punishment: "N/A" });
        }

        const data = doc.data();
   
        const completion = await groq.chat.completions.create({
            messages: [{
                role: "system",
                content: 'Return JSON only: {"punishment":"...", "cognizable":"Yes/No/NA", "bailable":"Yes/No/NA"}.'
            }, { role: "user", content: data.content }],
            model: "llama-3.3-70b-versatile", temperature: 0, response_format: { type: "json_object" }
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
    } catch (e) { res.status(500).json({ error: e.message }); }
});

router.post("/compare", async (req, res) => {
    try {
        const { act1, sec1, act2, sec2 } = req.body;
        const [snap1, snap2] = await Promise.all([
            db.collection("legal_sections").where("act_name", "==", act1).get(),
            db.collection("legal_sections").where("act_name", "==", act2).get()
        ]);

        const findMatch = (snap, s) => {
            const search = normalize(s);
            return snap.docs.find(d => normalize(d.data().section_number) === search || normalize(d.data().section_raw) === search);
        };

        const d1 = findMatch(snap1, sec1), d2 = findMatch(snap2, sec2);
        if (!d1 || !d2) return res.json({ formattedAnswer: "One or both sections not found." });

        const completion = await groq.chat.completions.create({
            messages: [{ role: "system", content: "Compare these laws. Output Markdown Table." }, { role: "user", content: `1: ${d1.data().content}\n2: ${d2.data().content}` }],
            model: "llama-3.3-70b-versatile", temperature: 0.1
        });
        res.json({ formattedAnswer: completion.choices[0].message.content });
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
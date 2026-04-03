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

const normalize = (str) => String(str || "").replace(/[^a-zA-Z0-9]/g, '').toLowerCase();


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
        console.log(`🔎 Lookup -> Act: ${act}, Section: ${section}`);

        // 🟢 THE FIX: Don't trust Firestore string matching alone. 
        // Use Pinecone to find the exact Document ID first.
        const searchQuery = `Act: ${act} Section: ${section}`;
        const queryVector = await getEmbedding(searchQuery);
        
        const searchResult = await index.namespace(NAMESPACE).query({
            vector: queryVector,
            topK: 20, // Grab more to find the best match
            includeMetadata: true
        });

        // Search for the ID that belongs to this Act and has this Section number
        const cleanReqAct = normalize(act);
        const cleanReqSec = normalize(section);

        const match = searchResult.matches.find(m => {
            const mAct = normalize(m.metadata.act);
            const mSec = normalize(m.metadata.section);
            // Verify it belongs to the act and section requested
            return (mAct.includes(cleanReqAct) || cleanReqAct.includes(mAct)) && mSec === cleanReqSec;
        });

        if (!match) {
            // Fallback: Direct Firestore Query if Vector search fails
            const snapshot = await db.collection("legal_sections")
                .where("section_number", "==", cleanReqSec)
                .limit(20).get();
            
            const fallbackDoc = snapshot.docs.find(d => normalize(d.data().act_name).includes(cleanReqAct));
            
            if (!fallbackDoc) {
                return res.json({ section: "!", title: "Not Found", description: "This specific section was not found in the selected Act.", punishment: "N/A" });
            }
            return sendSectionResponse(res, fallbackDoc.data());
        }

        const doc = await db.collection("legal_sections").doc(match.metadata.firestore_id).get();
        if (!doc.exists) throw new Error("Firestore ID mismatch");

        return sendSectionResponse(res, doc.data());

    } catch (e) { 
        console.error(e);
        res.status(500).json({ error: e.message }); 
    }
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
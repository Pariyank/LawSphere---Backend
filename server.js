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

let serviceAccount;
try {
    if (process.env.FIREBASE_SERVICE_ACCOUNT) {
        serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
    } else {
        serviceAccount = require("./firebase-service-account.json");
    }
    if (!admin.apps.length) {
        admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    }
} catch (e) {
    console.error("❌ Firebase Error:", e.message);
}
const db = admin.firestore();

const PORT = process.env.PORT || 3000;
const NAMESPACE = "default";
const LLM_MODEL = "llama-3.3-70b-versatile"; 

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX || "lawsphere-index");
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

let embedder = null;
async function loadModel() {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
}

async function getEmbedding(text) {
    if (!embedder) await loadModel();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data).map(Number);
}

const normalize = (str) => String(str).replace(/[^a-zA-Z0-9]/g, '').toLowerCase();

const router = express.Router();

router.get("/", (req, res) => res.send("🚀 LawSphere Engine Active"));

router.post("/ask", async (req, res) => {
    try {
        const { query, language } = req.body;
        const queryVector = await getEmbedding(query);

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
                content: `You are LawSphere AI. ${lang} Answer ONLY using provided Context. Use simple words. Cite Act and Section.`
            }, { role: "user", content: `CONTEXT:\n${contextText}\n\nQUESTION: ${query}` }],
            model: LLM_MODEL, temperature: 0.1
        });

        res.json({ formattedAnswer: completion.choices[0].message.content, retrievedSources: sources.slice(0, 5) });
    } catch (error) { res.status(500).json({ formattedAnswer: "Brain connection error." }); }
});

router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body;
        console.log(`🔎 Lookup -> Act: ${act}, SearchTerm: ${section}`);
        const snapshot = await db.collection("legal_sections")
            .where("act_name", "==", act)
            .get();

        if (snapshot.empty) {
            return res.json({ title: "Act Not Found", description: "This Act is not in the database." });
        }
        const searchNorm = normalize(section);
        
        const doc = snapshot.docs.find(d => {
            const data = d.data();
            const dbSecNumNorm = normalize(data.section_number || "");
            const dbSecRawNorm = normalize(data.section_raw || "");
       
            return dbSecNumNorm === searchNorm || 
                   dbSecRawNorm === searchNorm || 
                   dbSecNumNorm === `section${searchNorm}` ||
                   dbSecNumNorm === `article${searchNorm}`;
        });

        if (!doc) {
            return res.json({ 
                section: section, 
                title: "Section Not Found", 
                description: `We found the Act, but could not find '${section}' inside it. Please check the number.`, 
                punishment: "N/A" 
            });
        }

        const data = doc.data();
        console.log(`✅ Match Found: ${data.title}`);

        // 3. AI Extraction for UI Tags
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

    } catch (e) { 
        console.error("Lookup Error:", e);
        res.status(500).json({ description: "Lookup Error: " + e.message }); 
    }
});


router.post("/compare", async (req, res) => {
    try {
        const { act1, sec1, act2, sec2 } = req.body;
        console.log(`⚖️ Comparing: [${act1} - ${sec1}] VS [${act2} - ${sec2}]`);

        const cleanSec1 = normalize(sec1);
        const cleanSec2 = normalize(sec2);

        const [snap1, snap2] = await Promise.all([
            db.collection("legal_sections").where("act_name", "==", act1).get(),
            db.collection("legal_sections").where("act_name", "==", act2).get()
        ]);

        const findMatch = (snap, searchNorm) => snap.docs.find(d => {
            const data = d.data();
            return normalize(data.section_number || "") === searchNorm || 
                   normalize(data.section_raw || "") === searchNorm;
        });

        const doc1 = findMatch(snap1, cleanSec1);
        const doc2 = findMatch(snap2, cleanSec2);

        if (!doc1 || !doc2) {
            return res.json({ 
                formattedAnswer: `❌ **Error:** Could not find one or both sections.\n\n- Found ${act1} Sec ${sec1}: ${!!doc1}\n- Found ${act2} Sec ${sec2}: ${!!doc2}` 
            });
        }

        const data1 = doc1.data();
        const data2 = doc2.data();
        const completion = await groq.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content: "You are a Legal Analyst. Compare the two provided legal provisions based ONLY on the text. Provide a Markdown table with columns: Feature, Provision 1, Provision 2. Include: Title, Definition, and Punishment."
                },
                {
                    role: "user",
                    content: `PROVISION 1: (From ${act1})\n${data1.content}\n\nPROVISION 2: (From ${act2})\n${data2.content}`
                }
            ],
            model: LLM_MODEL,
            temperature: 0.1
        });

        res.json({ 
            formattedAnswer: completion.choices[0].message.content,
            semanticTags: ["Comparison", "Side-by-Side"] 
        });

    } catch (error) {
        console.error("Compare Error:", error);
        res.status(500).json({ formattedAnswer: "Comparison process failed." });
    }
});


app.use("/api", router);
app.listen(PORT, "0.0.0.0", async () => { await loadModel(); console.log(`🚀 Server on Port ${PORT}`); });
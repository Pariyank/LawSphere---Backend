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

// ================= 2. CONFIG & SERVICES =================
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

// Helper to clean strings for comparison (Removes everything except letters and numbers)
// Example: "Section 9." -> "section9" | "9" -> "9"
const normalize = (str) => String(str).replace(/[^a-zA-Z0-9]/g, '').toLowerCase();

// ================= 3. ROUTES =================
const router = express.Router();

router.get("/", (req, res) => res.send("🚀 LawSphere Engine Active"));

// ---------------------------------------------------------
// 🟢 ROUTE 1: CHATBOT (/api/ask)
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// 🟢 ROUTE 2: LIBRARY LOOKUP (FIXED NORMALIZATION)
// ---------------------------------------------------------
router.post("/lookup", async (req, res) => {
    try {
        const { act, section } = req.body;
        console.log(`🔎 Lookup -> Act: ${act}, SearchTerm: ${section}`);

        // 1. Get all entries for this specific Act from Firestore
        const snapshot = await db.collection("legal_sections")
            .where("act_name", "==", act)
            .get();

        if (snapshot.empty) {
            return res.json({ title: "Act Not Found", description: "This Act is not in the database." });
        }

        // 2. 🟢 SMART MATCHING LOGIC
        // We look through the results and normalize the numbers to find a match.
        // This handles "9" vs "Section 9" vs "Section9" vs "Article 9"
        const searchNorm = normalize(section);
        
        const doc = snapshot.docs.find(d => {
            const data = d.data();
            const dbSecNumNorm = normalize(data.section_number || "");
            const dbSecRawNorm = normalize(data.section_raw || "");
            
            // Match if "section9" contains "9" or if "9" equals "9"
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

router.get("/news", async (req, res) => {
    try {
        const apiKey = process.env.NEWS_API_KEY;
        const url = `https://gnews.io/api/v4/search?q=Supreme%20Court%20India&lang=en&country=in&max=10&apikey=${apiKey}`;
        const response = await axios.get(url);
        res.json(response.data.articles.map(a => ({ title: a.title, description: a.description, source: a.source.name, date: new Date(a.publishedAt).toDateString() })));
    } catch (error) { res.json([{ title: "News Unavailable", description: "Check API limit.", source: "System", date: "Now" }]); }
});

app.use("/api", router);
app.listen(PORT, "0.0.0.0", async () => { await loadModel(); console.log(`🚀 Server on Port ${PORT}`); });
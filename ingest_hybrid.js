require("dotenv").config();
const fs = require("fs"), path = require("path"), admin = require("firebase-admin");
const { Pinecone } = require("@pinecone-database/pinecone"), { pipeline } = require("@xenova/transformers");

// ================= SETUP FIREBASE ADMIN =================
const sa = require("./firebase-service-account.json");
if (!admin.apps.length) admin.initializeApp({ credential: admin.credential.cert(sa) });
const db = admin.firestore();

// ================= CONFIG =================
const IN_DIR = path.join(__dirname, "renamed_json"), BATCH = 20;
const PINECONE_KEY = process.env.PINECONE_API_KEY, INDEX_NAME = "lawsphere-index";
let embedder = null, host = "";

// ================= UTILITIES =================
const load = async () => { embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2"); };

const getVec = async (t) => {
    if (!embedder) await load();
    const out = await embedder(t, { pooling: "mean", normalize: true });
    return Array.from(out.data).map(Number);
};

const clean = s => s ? String(s).replace(/[^a-zA-Z0-9]/g, "") : "NA";

// 🟢 RECURSIVE TEXT EXTRACTOR (Handles Admiralty & complex nesting)
const ext = n => {
    if (!n) return "";
    if (typeof n === 'string') return n.trim();
    if (Array.isArray(n)) return n.map(ext).join("\n").trim();
    if (typeof n === 'object') {
        let p = [];
        if (n.text) p.push(n.text);
        if (n.contains) p.push(ext(n.contains));
        if (!n.text && !n.contains) {
            Object.keys(n).sort((a,b) => parseInt(a) - parseInt(b)).forEach(k => p.push(ext(n[k])));
        }
        return p.join("\n").trim();
    }
    return "";
};

// 🟢 DIRECT HTTP UPSERT (Fixes Node v24 SDK Bug)
const upsert = async (v) => {
    const url = `https://${host}/vectors/upsert`;
    const res = await fetch(url, {
        method: "POST",
        headers: { "Api-Key": PINECONE_KEY, "Content-Type": "application/json" },
        body: JSON.stringify({ vectors: v, namespace: "default" })
    });
    if (!res.ok) throw new Error(await res.text());
};

// ================= CORE LOGIC =================
async function processFile(fn) {
    console.log(`\n📄 Processing: ${fn}`);
    try {
        let raw = JSON.parse(fs.readFileSync(path.join(IN_DIR, fn), "utf8"));
        let act = fn.replace(".json", ""), q = [];
        const data = Array.isArray(raw) ? raw[0] : raw;

        // 1️⃣ SCHEMA: FLAT ARRAY (Constitution)
        if (Array.isArray(raw) && (raw[0].article !== undefined || raw[0].section !== undefined)) {
            console.log("   ➤ Logic: Flat Array");
            raw.forEach(i => {
                const isArt = i.article !== undefined;
                const num = isArt ? i.article : i.section;
                q.push({ cn: "General", sk: isArt ? `Article ${num}` : `Section ${num}`, sn: clean(num), h: i.title || "Law", t: ext(i.description || i.content) });
            });
        } 
        // 2️⃣ SCHEMA: NESTED OR FLAT OBJECT (Bare Acts)
        else {
            // A. Preamble
            if (data["Act Definition"]) q.push({ cn: "Preamble", sk: "Preamble", sn: "Preamble", h: "Definition", t: ext(data["Act Definition"]) });
            
            // B. Nested Chapters (Act -> Chapters -> Sections)
            if (data["Chapters"]) {
                console.log("   ➤ Logic: Nested Chapters");
                for (let k in data["Chapters"]) {
                    const c = data["Chapters"][k], secs = c.Sections || {};
                    for (let sk in secs) q.push({ cn: c.Name || k, sk: sk, sn: clean(sk.replace(/[^0-9A-Z]/ig, '')), h: secs[sk].heading || "Law", t: ext(secs[sk].paragraphs) });
                }
            }

            // C. Direct Sections (Act -> Sections directly)
            if (data["Sections"]) {
                console.log("   ➤ Logic: Direct Sections");
                for (let sk in data["Sections"]) {
                    const s = data["Sections"][sk];
                    q.push({ cn: "General", sk: sk, sn: clean(sk.replace(/[^0-9A-Z]/ig, '')), h: s.heading || "Law", t: ext(s.paragraphs) });
                }
            }

            // D. Extras (Schedules, Footnotes, etc)
            ["Schedule", "Annexure", "Appendix", "Forms", "Footnotes"].forEach(p => {
                if (data[p]) {
                    for (let k in data[p]) q.push({ cn: p, sk: k, sn: clean(k), h: `${p}: ${k}`, t: ext(data[p][k]) });
                }
            });
        }

        console.log(`   🧩 Items to store: ${q.length}`);

        let vecs = [], total = 0;
        for (let i = 0; i < q.length; i++) {
            const item = q[i]; if (!item.t || item.t.length < 5) continue;
            const docId = `${clean(act)}_${clean(item.cn)}_${item.sn}_${i}`;

            // 1. Save to Firestore (Truth)
            await db.collection("legal_sections").doc(docId).set({
                act_name: act, chapter_name: item.cn, section_raw: item.sk, section_number: item.sn, title: item.h, content: item.t
            });

            // 2. Save to Pinecone (Search)
            const embedding = await getVec(`Act: ${act} | ${item.sk}: ${item.h}\n${item.t}`);
            vecs.push({
                id: docId, values: embedding,
                metadata: { firestore_id: docId, act: act, section: item.sn, text: item.t.substring(0, 300) }
            });

            if (i % 5 === 0) process.stdout.write(".");
            if (vecs.length >= BATCH) { await upsert(vecs); total += vecs.length; vecs = []; }
        }
        if (vecs.length > 0) { await upsert(vecs); total += vecs.length; }
        console.log(`\n✅ Finished: ${total} records.`);
        return total;

    } catch (e) { console.error(`❌ Error in ${fn}: ${e.message}`); return 0; }
}

async function main() {
    console.log("------------------------------------------------");
    console.log("🚀 STARTING DUAL-SCHEMA INGESTION");
    console.log("------------------------------------------------");
    try {
        const pinecone = new Pinecone({ apiKey: PINECONE_KEY });
        const desc = await pinecone.describeIndex(INDEX_NAME);
        host = desc.host;
        await load();
        const files = fs.readdirSync(IN_DIR).filter(f => f.endsWith(".json"));
        let grandTotal = 0;
        for (const f of files) grandTotal += await processFile(f);
        console.log(`\n🎉 DONE! Total items in LawSphere: ${grandTotal}`);
    } catch (err) { console.error("FATAL:", err.message); }
}

main();
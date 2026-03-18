require("dotenv").config();
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const Groq = require("groq-sdk");

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const LLM_MODEL = "llama-3.3-70b-versatile"; 

const INPUT_DIR = path.join(__dirname, "new_raw_pdfs");
const OUTPUT_DIR = path.join(__dirname, "converted_json");

const delay = (ms) => new Promise(res => setTimeout(res, ms));

// 🟢 1. Extract Top-Level Metadata
async function extractMetadata(text) {
    const sample = text.substring(0, 4000);
    const prompt = `Extract metadata from this Indian Act. Return ONLY raw JSON:
    {
        "Act Title": "Full name in uppercase",
        "Act ID": "ACT NO. X OF YEAR",
        "Enactment Date": "[Date in brackets]",
        "Act Definition": { "0": "The 'An Act to...' preamble" }
    }`;

    const chat = await groq.chat.completions.create({
        messages: [{ role: "system", content: prompt }, { role: "user", content: sample }],
        model: LLM_MODEL, temperature: 0, response_format: { type: "json_object" }
    });
    return JSON.parse(chat.choices[0].message.content);
}

// 🟢 2. Extract Chapters & Sections with exact paragraph mapping
async function processChapter(chapterContent) {
    const prompt = `Convert the following legal text into JSON. 
    RULES:
    1. Identify Chapter ID (e.g., "CHAPTER I") and Name (e.g., "PRELIMINARY").
    2. Sections MUST be keys like "Section 1.", "Section 2.".
    3. Under each section, provide a "heading".
    4. "paragraphs" MUST be a dictionary with string indices "0", "1", "2"... 
    5. Include every word. Do not summarize. 
    
    JSON FORMAT:
    {
        "ID": "CHAPTER ...",
        "Name": "...",
        "Sections": {
            "Section 1.": {
                "heading": "...",
                "paragraphs": { "0": "(1) ...", "1": "(2) ..." }
            }
        }
    }`;

    const chat = await groq.chat.completions.create({
        messages: [{ role: "system", content: prompt }, { role: "user", content: chapterContent }],
        model: LLM_MODEL, temperature: 0, response_format: { type: "json_object" }
    });
    return JSON.parse(chat.choices[0].message.content);
}

async function convertFile(fileName) {
    console.log(`\n🚀 Processing: ${fileName}`);
    const dataBuffer = fs.readFileSync(path.join(INPUT_DIR, fileName));
    const pdfData = await pdfParse(dataBuffer);
    
    // Clean common PDF footer/header noise
    let fullText = pdfData.text.replace(/Page \d+ of \d+/g, "");
    fullText = fullText.replace(/THE GAZETTE OF INDIA EXTRAORDINARY/g, "");
    fullText = fullText.replace(/\s+/g, " ").trim();

    // Step A: Meta
    const resultJson = await extractMetadata(fullText);
    resultJson["Chapters"] = {};

    // Step B: Split into Chapters
    // Regex identifies "CHAPTER I", "CHAPTER II", etc.
    const chapterSplits = fullText.split(/(?=CHAPTER\s+[IVXLCDM]+)/gi);
    
    let actualChapters = chapterSplits.filter(c => c.toLowerCase().includes("chapter"));
    console.log(`   Found ${actualChapters.length} chapters.`);

    // Step C: Process Chapters
    for (let i = 0; i < actualChapters.length; i++) {
        console.log(`   ⏳ Formatting Chapter ${i + 1}...`);
        try {
            const chapterData = await processChapter(actualChapters[i].substring(0, 25000));
            resultJson["Chapters"][i.toString()] = chapterData;
            await delay(3000); // Respect Groq rate limits
        } catch (e) {
            console.error(`   ❌ Failed Chapter ${i + 1}: ${e.message}`);
        }
    }

    // Step D: Final Polish
    const finalWrapper = [resultJson];
    const outPath = path.join(OUTPUT_DIR, fileName.replace(".pdf", ".json"));
    fs.writeFileSync(outPath, JSON.stringify(finalWrapper, null, 4));
    console.log(`✅ Saved: ${outPath}`);
}

async function main() {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR);
    const files = fs.readdirSync(INPUT_DIR).filter(f => f.endsWith(".pdf"));
    for (const f of files) await convertFile(f);
}

main();
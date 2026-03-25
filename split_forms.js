const { PDFDocument } = require('pdf-lib');
const fs = require('fs');
const path = require('path');

// CONFIG
const INPUT_FILE_NAME = 'Bharatiya Nagarik Suraksha Sanhita, 2023 FORMS.pdf'; // 🟢 CHANGE THIS to your actual filename
const INPUT_PATH = path.join(__dirname, 'splitter_input', INPUT_FILE_NAME);
const OUTPUT_DIR = path.join(__dirname, 'splitter_output');

async function splitPdf() {
    console.log("------------------------------------------------");
    console.log("🚀 STARTING PDF SPLITTING PROCESS");
    console.log("------------------------------------------------");

    try {
        // 1. Load the master PDF
        if (!fs.existsSync(INPUT_PATH)) {
            console.error(`❌ Error: File not found at ${INPUT_PATH}`);
            return;
        }
        const existingPdfBytes = fs.readFileSync(INPUT_PATH);
        const masterPdf = await PDFDocument.load(existingPdfBytes);
        
        const totalPages = masterPdf.getPageCount();
        console.log(`📄 Master PDF loaded. Total pages found: ${totalPages}`);

        if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR);

        // 2. Loop through every page
        for (let i = 0; i < totalPages; i++) {
            // Create a new PDF document for this specific page
            const newPdf = await PDFDocument.create();
            
            // Copy the page from the master PDF
            const [copiedPage] = await newPdf.copyPages(masterPdf, [i]);
            newPdf.addPage(copiedPage);

            // 3. Serialize to bytes
            const pdfBytes = await newPdf.save();

            // 4. Save the individual file
            // Note: BNSS Forms start from No. 1, so we use i + 1
            const fileName = `Bharatiya Nagarik Suraksha Sanhita, 2023 Form_${i + 1}.pdf`;
            const filePath = path.join(OUTPUT_DIR, fileName);
            
            fs.writeFileSync(filePath, pdfBytes);
            console.log(`✅ Generated: ${fileName}`);
        }

        console.log("------------------------------------------------");
        console.log(`🎉 SUCCESS: ${totalPages} forms generated in 'splitter_output'`);
        console.log("------------------------------------------------");

    } catch (error) {
        console.error("❌ Fatal Error during splitting:", error.message);
    }
}

splitPdf();
const pdf = require('pdf-parse');

console.log("------------------------------------------------");
console.log("🔍 DEBUGGING PDF LIBRARY");
console.log("------------------------------------------------");
console.log("Type of library:", typeof pdf);
console.log("Structure:", pdf);
console.log("Keys:", Object.keys(pdf));

if (typeof pdf === 'function') {
    console.log("✅ It is a function! You can call pdf()");
} else if (pdf.default) {
    console.log("⚠️ It has a .default property. Use pdf.default()");
} else {
    console.log("❌ It is an object but I don't see the function.");
}
console.log("------------------------------------------------");
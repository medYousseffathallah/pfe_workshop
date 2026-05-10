const fs = require('fs');
const pdf = require('pdf-parse');

const dataBuffer = fs.readFileSync(String.raw`c:\Users\admin\Desktop\pfe_preparation\old_repports\maincorrectphase1.pdf`);

pdf(dataBuffer).then(function(data) {
    fs.writeFileSync('extracted_pdf.txt', data.text);
    console.log("Extraction complete.");
}).catch(function(error) {
    console.error("Error extracting PDF:", error);
});
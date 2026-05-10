const fs = require('fs');
const path = require('path');

const dir = 'c:/Users/admin/Desktop/pfe_preparation/main-thesis-source';
const chap5ZeroPath = path.join(dir, 'chap5_zeroshot.tex');
const chap5EvalPath = path.join(dir, 'chap5_evaluation.tex');
const chap5CombinedPath = path.join(dir, 'chap5_combined.tex');
const mainPath = path.join(dir, 'main.tex');

let zeroContent = fs.readFileSync(chap5ZeroPath, 'utf8');
let evalContent = fs.readFileSync(chap5EvalPath, 'utf8');

// We want to combine them under one Chapter 5: Results & Zero-Shot Deployment
// Let's take the evaluation sections and append them to the zero-shot content.
// But first, we need to fix the \chapter and \section headers to merge them smoothly.

// Remove the \chapter and \label from evalContent
evalContent = evalContent.replace(/\\chapter{.*?}\n/g, '');
evalContent = evalContent.replace(/\\label{.*?}\n/g, '');
// Change eval's Introduction to something else or remove it if redundant
evalContent = evalContent.replace(/\\section{Introduction}\nThis chapter provides a purely quantitative evaluation/g, '\\section{Quantitative Evaluation Metrics}\nThis section provides a purely quantitative evaluation');
// The rest can remain as sections or subsections.

// Change the main chapter title in zeroContent
zeroContent = zeroContent.replace(/\\chapter{.*?}/g, '\\chapter{Results \\& Zero-Shot Deployment}');
// Also make sure "Release 4" is replaced with "Phase 4" if not already done. (Though we checked this earlier).

let combinedContent = zeroContent + '\n\n' + evalContent;

fs.writeFileSync(chap5CombinedPath, combinedContent, 'utf8');

// Now update main.tex
let mainContent = fs.readFileSync(mainPath, 'utf8');
// main.tex currently includes chap5_evaluation.tex or chap5_zeroshot.tex. Let's replace whatever is there.
mainContent = mainContent.replace(/\\input{chap5_.*?\.tex}/g, ''); // remove any chap5 inputs
mainContent = mainContent.replace(/\\input{chap4_logistics\.tex}/g, '\\input{chap4_logistics.tex}\n\n\\input{chap5_combined.tex}');

fs.writeFileSync(mainPath, mainContent, 'utf8');
console.log("chap5_combined.tex created and main.tex updated.");
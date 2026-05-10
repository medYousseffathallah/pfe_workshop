const fs = require('fs');

const chap3Path = 'c:/Users/admin/Desktop/pfe_preparation/main-thesis-source/chap3_safety.tex';
let chap3Content = fs.readFileSync(chap3Path, 'utf8');

const newSotA = `\\section{Targeted State of the Art: The YOLO Family (v5 \\& v8)}
To satisfy the strict efficiency-to-accuracy ratio demanded by NPU deployment, this project leverages the YOLO (You Only Look Once) family of models. Large-scale transformer architectures, while highly accurate, exceed the memory and compute limits of edge devices. The YOLO family provides a spectrum of models where the Nano variants (YOLOv5n and YOLOv8n) offer the necessary balance for industrial applications.

\\subsection{Architectural Evolution for Efficiency}
The evolution from YOLOv5 to YOLOv8 introduced several architectural shifts directly impacting edge deployment.

\\subsubsection{Memory Efficiency: C3 vs. C2f Modules}
The backbone of these models relies on cross-stage partial bottlenecks. The key difference lies in how feature information flows through the module:

\\begin{itemize}
  \\item \\textbf{C3 (YOLOv5)}: splits the input into two branches, passes one through a sequence of Bottleneck blocks, and concatenates the outputs. Uses 3 convolutional layers.
  \\item \\textbf{C2f (YOLOv8)}: splits the input and passes it through Bottleneck blocks, but \\emph{concatenates the outputs of all intermediate Bottleneck stages} with the split input. Uses 2 convolutional layers.
\\end{itemize}

This C2f-versus-C3 difference is detailed in Figure~\\ref{fig:c2f-vs-c3}, which highlights the richer intermediate feature concatenation used by C2f.

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{diagrams/fig6_c2f_vs_c3.png}
\\caption{C2f versus C3 module comparison}
\\label{fig:c2f-vs-c3}
\\end{figure}

While C2f provides richer gradient flow, it concatenates intermediate outputs from all Bottleneck stages, increasing peak memory consumption during inference. This makes YOLOv5's C3 module more memory-efficient for ultra-edge NPU deployments.

\\subsubsection{Detection Head: Coupled vs. Decoupled Design}
Unlike YOLOv5's coupled head, YOLOv8 employs a decoupled design (Figure~\\ref{fig:decoupled-head}) with separate branches for classification and regression.

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{diagrams/fig7_decoupled_head.png}
\\caption{Decoupled versus coupled detection heads}
\\label{fig:decoupled-head}
\\end{figure}

The decoupled design addresses the conflict between classification and localization objectives. Separating the branches allows each to specialize, which is crucial for multi-class detection (like Fire and Smoke). However, for single-class detection (like Forklifts), YOLOv5's coupled head provides the necessary functionality with fewer parameters and FLOPs.

\\subsection{Synthesis: Computational Efficiency and Resource Constraints}
The most direct comparison concerns the computational footprint of the nano variants, which are the primary deployment targets for NPU-based systems.

\\begin{center}
\\renewcommand{\\arraystretch}{1.3}
\\begin{tabular}{|p{0.20\\textwidth}|p{0.14\\textwidth}|p{0.14\\textwidth}|p{0.14\\textwidth}|p{0.22\\textwidth}|}
\\hline
\\textbf{Metric} & \\textbf{YOLOv5n} & \\textbf{YOLOv8n} & \\textbf{Difference} & \\textbf{Advantage} \\\\
\\hline
Parameters & 1.76M & 3.2M & $-$45\\% & YOLOv5n \\\\
\\hline
FLOPs & 4.1B & 8.7B & $-$53\\% & YOLOv5n \\\\
\\hline
Model Size & 3.56MB & 6.3MB & $-$43\\% & YOLOv5n \\\\
\\hline
CPU Inference & 73.6ms & 80.4ms & $-$8.5\\% & YOLOv5n \\\\
\\hline
GPU Inference (T4) & 0.6ms & 0.99ms & $-$39\\% & YOLOv5n \\\\
\\hline
\\end{tabular}
\\captionof{table}{YOLOv5n versus YOLOv8n compute comparison}
\\label{tab:v5n-vs-v8n-compute}
\\end{center}

YOLOv5n requires 45\\% fewer parameters and 53\\% fewer FLOPs than YOLOv8n. The computational cost difference between YOLOv5 and YOLOv8 is illustrated by comparing FLOPs across matching variants in Figure~\\ref{fig:flops-comparison}.

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{diagrams/fig11_flops_comparison.png}
\\caption{FLOPs comparison of YOLOv5 and YOLOv8}
\\label{fig:flops-comparison}
\\end{figure}

\\textbf{Conclusion for EYE-D:} The selection between YOLOv5n and YOLOv8n was driven by aligning the model's architectural strengths with the task's complexity. YOLOv5n (Anchor-based, Coupled Head, lowest FLOPs) was selected for Forklifts, as it is optimal for rigid, single-class objects with predictable aspect ratios, maximizing efficiency. YOLOv8n (Anchor-free, Decoupled Head, C2f modules) was selected for Fire/Smoke, as it provides higher accuracy for dynamic, semi-transparent objects with unpredictable shapes, fully utilizing the NPU compute budget.`;

// Replace the SotA section in chap3
const sotaStart = chap3Content.indexOf('\\section{Targeted State of the Art: The YOLO Family (v5 \\& v8)}');
const sotaEnd = chap3Content.indexOf('\\section{Implementation: Fire and Smoke Detection}');

if (sotaStart !== -1 && sotaEnd !== -1) {
    chap3Content = chap3Content.substring(0, sotaStart) + newSotA + '\n\n' + chap3Content.substring(sotaEnd);
    fs.writeFileSync(chap3Path, chap3Content, 'utf8');
    console.log("Replaced SotA in chap3_safety.tex successfully.");
} else {
    console.log("Could not find the SotA section in chap3_safety.tex.");
}

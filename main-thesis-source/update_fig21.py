import re

with open('chap2_data.tex', 'r', encoding='utf-8') as f:
    content = f.read()

replacement = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=2cm and 2.5cm,
  box/.style={draw, rectangle, rounded corners, align=center, minimum width=3cm, minimum height=1cm, fill=blue!10},
  arrow/.style={thick, ->, >=stealth}
]
  \node[box] (raw) {Raw Capture};
  \node[box, right=of raw] (ls) {Label Studio\\(Manual Annotation)};
  \node[box, right=of ls] (robo) {Roboflow\\(Healthcheck \& Analytics)};
  
  \draw[arrow] (raw) -- (ls);
  \draw[arrow] (ls) -- (robo);
  \draw[arrow] (robo) -- ++(0,-1.5) -| node[pos=0.25, below] {Correction Loop (Anomalies)} (ls);
\end{tikzpicture}
\caption{Dual-Pipeline Architecture: Annotation-to-Analytics Loop.}
\label{fig:dual-pipeline}
\end{figure}"""

# Replace the figure containing crisp_dm.png
content = re.sub(r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=0\.85\\textwidth\]\{diagrams/crisp_dm\.png\}\s*\\caption\{CRISP-DM Cycle Diagram highlighting the iterative methodology and data loops within Phase 1\.\}\s*\\label\{fig:crisp-dm-phase1\}\s*\\end\{figure\}', replacement, content)

with open('chap2_data.tex', 'w', encoding='utf-8') as f:
    f.write(content)

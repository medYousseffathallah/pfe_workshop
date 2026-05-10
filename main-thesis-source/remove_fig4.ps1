$content = Get-Content "chap4_logistics.tex" -Raw -Encoding UTF8

$content = [regex]::Replace($content, '(?s)\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=0\.8\\textwidth\]\{images/integration_architecture\.png\}\s*\\caption\{UML Integration Architecture showing the interaction between the Video Ingestion module, the NPU Inference Engine, and the Dashboard\.\}\s*\\label\{fig:integration_arch\}\s*\\end\{figure\}', '')

Set-Content "chap4_logistics.tex" -Value $content -Encoding UTF8

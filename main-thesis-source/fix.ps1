$main = Get-Content "main.tex" -Raw -Encoding UTF8
$main = $main -replace '\\usepackage\[hidelinks\]\{hyperref\}', "\usepackage{hyperref}`n\hypersetup{colorlinks=true, linkcolor=blue, citecolor=green, filecolor=magenta, urlcolor=cyan}"
$main = $main -replace '—', '-'
Set-Content "main.tex" -Value $main -Encoding UTF8

$chap2 = Get-Content "chap2_data.tex" -Raw -Encoding UTF8
$chap2 = $chap2 -replace '(?s)\\begin\{figure\}\[htbp\].*?\\label\{fig:occlusion-challenge\}.*?\\end\{figure\}', ''
$chap2 = $chap2 -replace '(?s)\\begin\{figure\}\[htbp\].*?\\label\{fig:illumination-challenge\}.*?\\end\{figure\}', ''
$chap2 = $chap2 -replace '—', '-'
Set-Content "chap2_data.tex" -Value $chap2 -Encoding UTF8

$chap4 = Get-Content "chap4_logistics.tex" -Raw -Encoding UTF8
$chap4 = $chap4 -replace '—', '-'

$forklift_impl_match = [regex]::match($chap4, '(?s)(\\section\{Implementation: Forklift Detection\}.*?)(?=\\section\{Implementation: Velocity \\& Orientation\})')
$forklift_impl_text = $forklift_impl_match.Groups[1].Value

$results_match = [regex]::match($chap4, '(?s)(\\section\{Results \\& Evaluation\}.*?)(?=\\section\{Chapter Summary\})')
$forklift_results_text = $results_match.Groups[1].Value

$chap4 = $chap4.Replace($forklift_impl_text, '')
$chap4 = $chap4.Replace($forklift_results_text, '')
Set-Content "chap4_logistics.tex" -Value $chap4 -Encoding UTF8

$chap3 = Get-Content "chap3_safety.tex" -Raw -Encoding UTF8
$chap3 = $chap3 -replace '—', '-'
$chap3 = $chap3.Replace('\section{Results \& Evaluation}', $forklift_impl_text + "`n\section{Results \& Evaluation}")

$forklift_results_subsections = [regex]::Replace($forklift_results_text, '(?s)\\section\{Results \\& Evaluation\}\\label\{sec:chap4-results\}.*?(?=\\subsection)', '')
$chap3 = $chap3.Replace('\section{Chapter Summary}', $forklift_results_subsections + "`n\section{Chapter Summary}")
Set-Content "chap3_safety.tex" -Value $chap3 -Encoding UTF8

foreach ($file in @("chap1.tex", "chap5_zeroshot.tex", "chap5_combined.tex")) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw -Encoding UTF8
        $content = $content -replace '—', '-'
        Set-Content $file -Value $content -Encoding UTF8
    }
}
Write-Host "Done"

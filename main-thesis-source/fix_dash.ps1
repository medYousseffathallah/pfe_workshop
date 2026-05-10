$files = Get-ChildItem *.tex
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    $content = $content.Replace([char]8212, "-").Replace([char]8211, "-")
    Set-Content $file.FullName -Value $content -Encoding UTF8
}
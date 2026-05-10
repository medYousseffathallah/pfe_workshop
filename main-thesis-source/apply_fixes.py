import re

# 1. Update main.tex
with open('main.tex', 'r', encoding='utf-8') as f:
    main_content = f.read()

# Replace \usepackage[hidelinks]{hyperref} with \usepackage{hyperref} and \hypersetup{...}
main_content = re.sub(
    r'\\usepackage\[hidelinks\]\{hyperref\}',
    r'\\usepackage{hyperref}\n\\hypersetup{colorlinks=true, linkcolor=blue, citecolor=green, filecolor=magenta, urlcolor=cyan}',
    main_content
)

# Global watermark replacement (replace em-dash with hyphen, but only the specific ones that are artifacts)
# "A global "Find and Replace" must be executed for the character —."
main_content = main_content.replace('—', '-')

with open('main.tex', 'w', encoding='utf-8') as f:
    f.write(main_content)

# 2. Update chap2_data.tex
with open('chap2_data.tex', 'r', encoding='utf-8') as f:
    chap2_content = f.read()

# Remove fig:occlusion-challenge
chap2_content = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:occlusion-challenge\}.*?\\end\{figure\}', '', chap2_content, flags=re.DOTALL)
# Remove fig:illumination-challenge
chap2_content = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:illumination-challenge\}.*?\\end\{figure\}', '', chap2_content, flags=re.DOTALL)

chap2_content = chap2_content.replace('—', '-')

with open('chap2_data.tex', 'w', encoding='utf-8') as f:
    f.write(chap2_content)

# 3. Read chap4_logistics.tex to extract Forklift logic
with open('chap4_logistics.tex', 'r', encoding='utf-8') as f:
    chap4_content = f.read()

chap4_content = chap4_content.replace('—', '-')

# Extract Section: Implementation: Forklift Detection
forklift_impl_match = re.search(r'(\\section\{Implementation: Forklift Detection\}.*?)(?=\\section\{Implementation: Velocity \\& Orientation\})', chap4_content, flags=re.DOTALL)
forklift_impl_text = forklift_impl_match.group(1) if forklift_impl_match else ''

# Extract Results & Evaluation section from Chap 4
results_match = re.search(r'(\\section\{Results \\& Evaluation\}.*?)(?=\\section\{Chapter Summary\})', chap4_content, flags=re.DOTALL)
forklift_results_text = results_match.group(1) if results_match else ''

# Remove them from chap4_logistics.tex
chap4_content = chap4_content.replace(forklift_impl_text, '')
chap4_content = chap4_content.replace(forklift_results_text, '')

# We need to make sure chap4_logistics.tex still has a section for Results if necessary, but actually the user said:
# "This chapter should now focus exclusively on the Mathematical Intelligence added on top of detection"
# So removing Results & Evaluation from Chap 4 is fine, or we can just leave it out.

with open('chap4_logistics.tex', 'w', encoding='utf-8') as f:
    f.write(chap4_content)

# 4. Update chap3_safety.tex
with open('chap3_safety.tex', 'r', encoding='utf-8') as f:
    chap3_content = f.read()

chap3_content = chap3_content.replace('—', '-')

# Insert Forklift Implementation before \section{Results \& Evaluation}
chap3_content = chap3_content.replace('\\section{Results \\& Evaluation}', forklift_impl_text + '\n\\section{Results \\& Evaluation}')

# We need to merge the forklift results into Chap 3's Results section.
# Chap 3's Results section ends at \section{Chapter Summary}
# We can just append the subsections from forklift_results_text to Chap 3's Results section.
# forklift_results_text contains \section{Results \& Evaluation}, so we should strip that header.

forklift_results_subsections = re.sub(r'\\section\{Results \\& Evaluation\}\\label\{sec:chap4-results\}.*?(?=\\subsection)', '', forklift_results_text, flags=re.DOTALL)

# Insert the forklift results right before \section{Chapter Summary} in Chap 3
chap3_content = chap3_content.replace('\\section{Chapter Summary}', forklift_results_subsections + '\n\\section{Chapter Summary}')

with open('chap3_safety.tex', 'w', encoding='utf-8') as f:
    f.write(chap3_content)

# Update chap5_zeroshot.tex and chap1.tex for em-dash as well
for filename in ['chap5_zeroshot.tex', 'chap1.tex', 'chap5_combined.tex']:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('—', '-')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    except:
        pass

print("Python script completed.")

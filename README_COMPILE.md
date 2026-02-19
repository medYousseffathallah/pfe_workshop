# PFE Report Compilation Guide

## Option 1: Online (Easiest)

1. Go to [Overleaf](https://www.overleaf.com/).
2. Create a new project.
3. Upload the following files from your folder:
   - `main.tex`
   - The entire `extracted_images` folder.
4. Click **Recompile**.
5. Download the PDF.

## Option 2: Local Installation

You need to install a LaTeX distribution to run the `xelatex` command.

- **Windows**: Install [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/).
- **After Installation**: Restart your computer/terminal and run:
  ```bash
  xelatex main.tex
  ```

## Option 3: Microsoft Word (DOCX)

If you prefer Word, use the content in `front_page_content.txt` and follow the formatting instructions below:

1. **Top-left logo**: Insert `extracted_images/Im1.jpg` near the upper-left corner.
2. **Top-right logo**: Insert `extracted_images/Im2.png` near the upper-right corner.
3. **Middle band**: Insert `extracted_images/Im3.png` as a full-width band around the middle top area.
4. **Order number**: Place `Order N°: L15` on the right side over the band.

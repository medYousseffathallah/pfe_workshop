---
name: "pfe-titlepage"
description: "Generates a PFE title page matching a reference PDF. Invoke when user asks to replicate/convert a PFE cover page to LaTeX/DOCX with high visual similarity."
---

# PFE Title Page

## Purpose

Generate a title page that matches a reference PFE PDF as closely as possible, using the existing project layout workflow.

## Invoke when

- The user asks to replicate a PFE report cover/title page.
- The user provides a reference PDF and expects a visually similar LaTeX output.
- The user asks to extract the title page content and format it for DOCX.

## Inputs

- Reference PDF path (typically under `old_repports/`).
- Desired output(s): LaTeX (`main.tex` + `main.pdf`), and optionally DOCX-ready text.

## Output

- `main.tex` and compiled `main.pdf`.
- `front_page_content.txt` (DOCX-ready content + formatting instructions).
- `titlepage_analysis.json` describing reference vs generated layout.

## Procedure

### 1) Locate the reference and assets

- Prefer the newest reference PDF.
- If needed, extract images from page 1 into `extracted_images/`.
- Reuse existing extracted assets when they already exist.

### 2) Analyze the reference title page

Run:

```bash
python analyze_titlepage.py
```

Use `titlepage_analysis.json`:

- `reference.blocks[].bbox` for text placement.
- `reference.images[].rects` for image placement.

### 3) Build a high-similarity LaTeX title page

Use this approach:

- `geometry` with zero margins.
- TikZ overlay nodes positioned in points using the reference coordinates.
- `fontspec` with Times New Roman.
- Place images at the exact rect coordinates.

Compile with:

```bash
xelatex -interaction=nonstopmode main.tex
```

### 4) Validate similarity

Run the analyzer again and compare `reference` vs `generated` blocks/images.

```bash
python analyze_titlepage.py
```

Iterate by adjusting coordinates until the block bounding boxes are close.

### 5) Produce DOCX-ready text (optional)

Update `front_page_content.txt`:

- Keep content exactly as the reference (including separators like `*****`).
- Add clear manual instructions for Word placement of logos/band/order number.

## Environment notes (Windows)

- Prefer `xelatex`.
- If LaTeX is missing, install MiKTeX via winget:

```bash
winget install -e --id MiKTeX.MiKTeX --accept-package-agreements
```

- If a new terminal can’t find `xelatex`, refresh PATH within that terminal session.


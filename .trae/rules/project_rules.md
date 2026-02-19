# Project Rules

## Goal

This workspace focuses on generating a PFE report title page that matches a reference PDF.

## When the user asks for a PFE title page

1. Identify the reference PDF (usually in `old_repports/`).
2. Extract page-1 text and images.
   - Prefer keeping extracted assets in `extracted_images/`.
   - Keep the same naming convention unless the user requests renaming.
3. Treat this as a layout-replication task.
   - Prefer absolute positioning using TikZ overlay and coordinates.
   - Do not “approximate” with generic centered blocks when the user wants similarity.

## Canonical workflow for similarity

1. Run `python analyze_titlepage.py` to regenerate `titlepage_analysis.json`.
2. Use the `reference.images[].rects` and `reference.blocks[].bbox` values to drive layout.
3. Build `main.tex` using:
   - `geometry` with zero margins for absolute positioning.
   - `tikz` with `remember picture, overlay`.
   - `fontspec` + `\setmainfont{Times New Roman}`.
4. Compile with XeLaTeX:
   - `xelatex -interaction=nonstopmode main.tex`

## Image mapping

When the reference comes from `Rapport PFE Khalyl Ebdelli.pdf`:

- Top-left logo: `extracted_images/Im1.jpg`
- Top-right logo: `extracted_images/Im2.png`
- Middle band: `extracted_images/Im3.png`

## Environment assumptions (Windows)

- Prefer `xelatex` over `pdflatex`.
- If LaTeX is missing, install MiKTeX via `winget`.
- If `xelatex` is not found in a new terminal session, refresh PATH within the session.

## Deliverables

- `main.tex` must compile to `main.pdf`.
- Keep `README_COMPILE.md` in sync with the actual compilation command.
- Keep `front_page_content.txt` aligned with the latest layout decisions.

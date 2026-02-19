import json
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parent


def image_info(path: Path):
    if not path.exists():
        return None
    with Image.open(path) as im:
        return {
            "path": str(path),
            "format": im.format,
            "mode": im.mode,
            "width": im.size[0],
            "height": im.size[1],
        }


def analyze_pdf(pdf_path: Path, page_index: int = 0):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)

    blocks = []
    for (x0, y0, x1, y1, text, block_no, block_type) in page.get_text("blocks"):
        cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        if not cleaned:
            continue
        blocks.append(
            {
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "text": cleaned,
                "block_type": block_type,
            }
        )

    images = []
    for img in page.get_images(full=True):
        xref = img[0]
        rects = page.get_image_rects(xref)
        images.append(
            {
                "xref": xref,
                "rects": [
                    [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
                    for r in rects
                ],
            }
        )

    return {
        "pdf": str(pdf_path),
        "page_index": page_index,
        "page_size": [round(page.rect.width, 2), round(page.rect.height, 2)],
        "blocks": blocks,
        "images": images,
    }


def main():
    reference_pdf = ROOT / "old_repports" / "Rapport PFE Khalyl Ebdelli.pdf"
    generated_pdf = ROOT / "main.pdf"

    extracted_dir = ROOT / "extracted_images"
    extracted = []
    if extracted_dir.exists():
        for p in sorted(extracted_dir.iterdir()):
            if p.is_file():
                info = image_info(p)
                if info:
                    extracted.append(info)

    out = {
        "extracted_images": extracted,
        "reference": analyze_pdf(reference_pdf, 0),
        "generated": analyze_pdf(generated_pdf, 0),
    }

    out_path = ROOT / "titlepage_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()


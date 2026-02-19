import pathlib
import re
import zipfile
import xml.etree.ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def _local_name(tag: str) -> str:
    if "}" not in tag:
        return tag
    return tag.split("}", 1)[1]


def _iter_text(node: ET.Element) -> str:
    parts: list[str] = []
    for t in node.findall(".//w:t", NS):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _read_xml_from_docx(z: zipfile.ZipFile, path: str) -> ET.Element | None:
    try:
        data = z.read(path)
    except KeyError:
        return None
    return ET.fromstring(data)


def extract_comments(z: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    root = _read_xml_from_docx(z, "word/comments.xml")
    if root is None:
        return {}

    comments: dict[str, dict[str, str]] = {}
    for c in root.findall(".//w:comment", NS):
        cid = c.attrib.get(f"{{{W_NS}}}id")
        if cid is None:
            continue
        author = c.attrib.get(f"{{{W_NS}}}author", "")
        date = c.attrib.get(f"{{{W_NS}}}date", "")
        text = _iter_text(c).strip()
        comments[cid] = {"author": author, "date": date, "text": text}
    return comments


def extract_doc_paragraphs_with_comment_ids(
    z: zipfile.ZipFile,
) -> list[dict[str, object]]:
    root = _read_xml_from_docx(z, "word/document.xml")
    if root is None:
        return []

    paragraphs: list[dict[str, object]] = []
    for p in root.findall(".//w:p", NS):
        style_val = ""
        ppr = p.find("./w:pPr", NS)
        if ppr is not None:
            pstyle = ppr.find("./w:pStyle", NS)
            if pstyle is not None:
                style_val = pstyle.attrib.get(f"{{{W_NS}}}val", "")

        text = _iter_text(p).strip()
        if not text:
            continue

        comment_ids: set[str] = set()
        for marker in p.findall(".//w:commentRangeStart", NS):
            cid = marker.attrib.get(f"{{{W_NS}}}id")
            if cid is not None:
                comment_ids.add(cid)
        for marker in p.findall(".//w:commentReference", NS):
            cid = marker.attrib.get(f"{{{W_NS}}}id")
            if cid is not None:
                comment_ids.add(cid)

        paragraphs.append(
            {"text": text, "comment_ids": sorted(comment_ids), "style": style_val}
        )
    return paragraphs


def extract_tracked_changes(z: zipfile.ZipFile) -> list[dict[str, str]]:
    root = _read_xml_from_docx(z, "word/document.xml")
    if root is None:
        return []

    changes: list[dict[str, str]] = []

    for parent in root.findall(".//w:p", NS):
        p_text = _iter_text(parent).strip()
        if not p_text:
            continue

        for ins in parent.findall(".//w:ins", NS):
            itext = _iter_text(ins).strip()
            if itext:
                changes.append({"type": "insertion", "context": p_text, "text": itext})

        for dele in parent.findall(".//w:del", NS):
            dtext_parts: list[str] = []
            for dt in dele.findall(".//w:delText", NS):
                if dt.text:
                    dtext_parts.append(dt.text)
            dtext = "".join(dtext_parts).strip()
            if dtext:
                changes.append({"type": "deletion", "context": p_text, "text": dtext})

    return changes


def main() -> None:
    docx_path = pathlib.Path("pfe_version0_15_02_2026.docx")
    out_path = pathlib.Path("supervisor_notes_extracted.txt")

    with zipfile.ZipFile(docx_path) as z:
        comments = extract_comments(z)
        paragraphs = extract_doc_paragraphs_with_comment_ids(z)
        changes = extract_tracked_changes(z)

    lines: list[str] = []
    lines.append("SOURCE_DOCX=" + str(docx_path))
    lines.append("")

    if not comments:
        lines.append("COMMENTS_FOUND=0")
    else:
        lines.append("COMMENTS_FOUND=" + str(len(comments)))
        lines.append("")
        lines.append("COMMENTS:")
        for cid in sorted(comments.keys(), key=lambda x: int(x) if x.isdigit() else 10**9):
            c = comments[cid]
            lines.append("")
            lines.append("  - id=" + cid)
            if c["author"]:
                lines.append("    author=" + c["author"])
            if c["date"]:
                lines.append("    date=" + c["date"])
            lines.append("    text=" + (c["text"] or "(empty)"))

    lines.append("")
    lines.append("PARAGRAPHS_WITH_COMMENT_REFERENCES:")
    any_ref = False
    for p in paragraphs:
        ids = p["comment_ids"]
        if ids:
            any_ref = True
            lines.append("")
            lines.append("  - comment_ids=" + ",".join(ids))
            if p["style"]:
                lines.append("    style=" + str(p["style"]))
            lines.append("    paragraph=" + str(p["text"]))
    if not any_ref:
        lines.append("")
        lines.append("  (none)")

    lines.append("")
    lines.append("TRACKED_CHANGES:")
    if not changes:
        lines.append("")
        lines.append("  (none)")
    else:
        for ch in changes:
            lines.append("")
            lines.append("  - type=" + ch["type"])
            lines.append("    text=" + ch["text"])
            lines.append("    context=" + ch["context"])

    lines.append("")
    lines.append("POTENTIAL_SUPERVISOR_NOTES:")
    patterns = [
        re.compile(r"\?\?"),
        re.compile(r"\bTODO\b", re.IGNORECASE),
        re.compile(r"\bFIXME\b", re.IGNORECASE),
        re.compile(r"\bNOTE\b", re.IGNORECASE),
    ]
    any_notes = False
    for idx, p in enumerate(paragraphs, start=1):
        text = str(p["text"])
        if any(pt.search(text) for pt in patterns):
            any_notes = True
            lines.append("")
            lines.append("  - paragraph_index=" + str(idx))
            if p["style"]:
                lines.append("    style=" + str(p["style"]))
            lines.append("    paragraph=" + text)
    if not any_notes:
        lines.append("")
        lines.append("  (none)")

    lines.append("")
    lines.append("FULL_PLAIN_TEXT:")
    for idx, p in enumerate(paragraphs, start=1):
        style = str(p["style"]) if p["style"] else ""
        style_suffix = (" [" + style + "]") if style else ""
        lines.append("")
        lines.append(str(idx) + style_suffix + " " + str(p["text"]))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

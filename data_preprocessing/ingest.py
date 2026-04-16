from pathlib import Path
import pymupdf4llm
import pdfplumber

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "document_files"
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


def _table_to_markdown(table: list) -> str:
    if not table or not table[0]:
        return ""
    rows = [[str(cell) if cell is not None else "" for cell in row] for row in table]
    header = rows[0]
    body = rows[1:]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_tables(pdf_path: str) -> dict:
    table_map = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue
            blocks = [_table_to_markdown(t) for t in tables if t]
            blocks = [b for b in blocks if b]
            if blocks:
                table_map[page_num] = "\n\n".join(blocks)
    return table_map


def load_pdf(file_path: str) -> list:
    full_path = str(DATA_DIR / file_path)
    pages_md = pymupdf4llm.to_markdown(
        full_path,
        page_chunks=True,
        write_images=True,
        image_path=str(ASSETS_DIR),
    )

    table_map = _extract_tables(full_path)

    result = []
    for page in pages_md:
        meta = page.get("metadata", {})
        page_num = meta.get("page_number") or (meta.get("page", 0) + 1)

        text = page.get("text", "")
        tables_md = table_map.get(page_num, "")
        has_tables = bool(tables_md)
        if has_tables:
            text = text + "\n\n### Tables\n\n" + tables_md

        images = []
        for img in page.get("images", []):
            images.append(img.get("path", img) if isinstance(img, dict) else img)

        result.append({
            "text": text,
            "metadata": {
                "file_path": file_path,
                "page_number": page_num,
                "has_images": bool(images),
                "has_tables": has_tables,
            },
            "images": images,
        })

    return result

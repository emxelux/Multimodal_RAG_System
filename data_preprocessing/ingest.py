"""
ingest.py — PDF loading with text, image, and table support.

Strategy:
  - pymupdf4llm  → converts each page to Markdown (preserves headings, bold,
                    inline images written to disk)
  - pdfplumber   → extracts structured tables per page, appended as Markdown tables
"""

from pathlib import Path
import pymupdf4llm
import pdfplumber

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "document_files"
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table (list-of-lists) to a Markdown table string."""
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
    """Return {1-based page number: markdown table string} for pages that have tables."""
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


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> list:
    """
    Load a PDF and return a list of page dicts, each containing:
        text      — Markdown text for the page (includes serialised tables)
        metadata  — dict: file_path, page_number, has_images, has_tables
        images    — list of image file paths written to assets/ (may be empty)

    Args:
        file_path: filename relative to DATA_DIR, e.g. "report.pdf"
    """
    full_path = str(DATA_DIR / file_path)

    # pymupdf4llm → Markdown per page, images saved to ASSETS_DIR
    pages_md = pymupdf4llm.to_markdown(
        full_path,
        page_chunks=True,
        write_images=True,
        image_path=str(ASSETS_DIR),
    )

    # pdfplumber → tables per page
    table_map = _extract_tables(full_path)

    result = []
    for page in pages_md:
        meta = page.get("metadata", {})
        # pymupdf4llm page numbering varies by version — normalise to 1-based
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

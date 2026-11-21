
from pathlib import Path
import pymupdf


def load_pdf(pdf_path: str) -> list[dict]:
    """Charger un PDF et extraire son texte page par page."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {pdf_path}"
        )

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number
                }
            })

    return pages

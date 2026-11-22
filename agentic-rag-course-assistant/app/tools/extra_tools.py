
import re

from app.ingestion.pdf_loader import load_pdf
from app.tools.web_search import web_search


PDF_PATH = "data/cours_transformers.pdf"


def document_lookup(question: str):
    """Lire une page explicitement demandée."""

    match = re.search(
        r"\b(?:page\s*|p\.\s*)(\d+)\b",
        question,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    page_number = int(match.group(1))
    pages = load_pdf(PDF_PATH)

    for page in pages:
        if page["metadata"]["page"] == page_number:
            return {
                "text": page["text"],
                "source": page["metadata"]["source"],
                "page": page_number
            }

    return None



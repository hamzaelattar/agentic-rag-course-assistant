
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunking import create_chunks


PDF_PATH = "data/cours_transformers.pdf"

# Charger le PDF
pages = load_pdf(PDF_PATH)

print("Nombre de pages :", len(pages))

# Découper le cours en chunks
chunks = create_chunks(pages)

print("Nombre de chunks :", len(chunks))

# Afficher les 3 premiers chunks
for chunk in chunks[:3]:
    print("\n" + "=" * 50)
    print("ID :", chunk["id"])
    print("Page :", chunk["metadata"]["page"])
    print("Index :", chunk["metadata"]["chunk_index"])
    print("Texte :\n", chunk["text"])

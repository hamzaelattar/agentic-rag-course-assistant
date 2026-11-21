
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages: list[dict]) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        text_chunks = splitter.split_text(text)

        for index, chunk_text in enumerate(text_chunks):

            unique_key = (
                f"{metadata['source']}:"
                f"{metadata['page']}:"
                f"{index}:{chunk_text}"
            )

            chunk_id = hashlib.sha256(
                unique_key.encode("utf-8")
            ).hexdigest()[:16]

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_index": index
                }
            })

    return chunks

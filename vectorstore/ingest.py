from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from tools.retriever import get_collection


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def iter_documents(data_dir: Path):
    for path in data_dir.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        category = path.relative_to(data_dir).parts[0] if len(path.relative_to(data_dir).parts) > 1 else "general"
        text = read_file(path)
        for index, chunk in enumerate(chunk_text(text)):
            digest = hashlib.sha1(f"{path}:{index}:{chunk[:80]}".encode("utf-8")).hexdigest()
            yield digest, chunk, {"source": str(path), "category": category, "chunk": index}


def main() -> None:
    data_dir = ROOT / os.getenv("DATA_DIR", "data")
    collection = get_collection()
    ids, docs, metadatas = [], [], []
    for doc_id, content, metadata in iter_documents(data_dir):
        ids.append(doc_id)
        docs.append(content)
        metadatas.append(metadata)

    if not ids:
        print(f"No .txt, .md, or .pdf files found in {data_dir}")
        return

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Ingested {len(ids)} chunks into Chroma collection 'insurance_documents'.")


if __name__ == "__main__":
    main()

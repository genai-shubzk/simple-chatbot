"""
prepare_data.py — Run this ONCE locally before deploying to Vercel.

What it does:
  1. Loads all .docx files from ../sample_docs/
  2. Chunks them (500 chars, 50 overlap)
  3. Embeds every chunk with OpenAI text-embedding-3-small
  4. Saves chunks + embeddings to data/embeddings.json

The output file is committed to the repo. On Vercel, api/chat.py loads it
into an in-memory ChromaDB — no filesystem writes needed at runtime.

Usage:
    pip install -r requirements-dev.txt
    export OPENAI_API_KEY=sk-...
    python prepare_data.py
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_FOLDER   = str(Path(__file__).parent.parent / "sample_docs")
OUTPUT_FILE   = Path(__file__).parent / "data" / "embeddings.json"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set.")

    # 1. Load documents
    print(f"Loading documents from {DOCS_FOLDER} ...")
    loader = DirectoryLoader(DOCS_FOLDER, glob="*.docx", loader_cls=Docx2txtLoader)
    docs = loader.load()
    print(f"  Loaded {len(docs)} documents")

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    # 3. Embed all chunks in one batched API call
    print("Embedding with OpenAI text-embedding-3-small ...")
    model  = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    texts  = [c.page_content for c in chunks]
    vectors = model.embed_documents(texts)
    print(f"  Embedded {len(vectors)} chunks")

    # 4. Save to JSON
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    data = [
        {
            "id":        str(i),
            "text":      texts[i],
            "source":    Path(chunks[i].metadata.get("source", "")).name,
            "embedding": vectors[i],
        }
        for i in range(len(chunks))
    ]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    size_mb = OUTPUT_FILE.stat().st_size / 1_000_000
    print(f"\nSaved {len(data)} chunks to {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print("Next step: git add data/embeddings.json && git commit && git push")


if __name__ == "__main__":
    main()

"""
reindex.py — Run this ONCE locally before deploying to Vercel.

What it does:
  1. Loads all .docx files from ../sample_docs/
  2. Chunks them
  3. Embeds with OpenAI text-embedding-3-small  (replaces HuggingFace)
  4. Saves a fresh ChromaDB to ./chroma_db/

After running this, commit the chroma_db/ folder to your repo.
Vercel will bundle it as read-only storage for the serverless function.

Usage:
    pip install -r requirements-dev.txt
    export OPENAI_API_KEY=sk-...
    python reindex.py
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_FOLDER    = str(Path(__file__).parent.parent / "sample_docs")
CHROMA_DIR     = str(Path(__file__).parent / "chroma_db")
COLLECTION     = "technova_policies"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set.")

    # 1. Wipe existing DB so we start clean
    if Path(CHROMA_DIR).exists():
        print(f"Removing existing chroma_db at {CHROMA_DIR} ...")
        shutil.rmtree(CHROMA_DIR)

    # 2. Load documents
    print(f"Loading documents from {DOCS_FOLDER} ...")
    loader = DirectoryLoader(DOCS_FOLDER, glob="*.docx", loader_cls=Docx2txtLoader)
    docs = loader.load()
    print(f"  Loaded {len(docs)} documents")

    # 3. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    # 4. Embed and store (uses OpenAI — no local model download)
    print("Embedding with OpenAI text-embedding-3-small ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    print(f"  Stored {vectorstore._collection.count()} chunks in {CHROMA_DIR}")
    print("\nDone! Now run:  git add chroma_db  and commit before deploying.")


if __name__ == "__main__":
    main()

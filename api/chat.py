"""
Vercel Python Serverless Function — POST /api/chat

Loads pre-computed embeddings from data/embeddings.json into an in-memory
ChromaDB instance. No filesystem writes — works on Vercel's read-only runtime.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_FILE       = Path(__file__).parent.parent / "data" / "embeddings.json"
COLLECTION_NAME = "technova_policies"

# ── Prompts ───────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in our policy documents."
Be concise and specific.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")

NO_RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question about company HR, IT, and Finance policies.

QUESTION: {question}

ANSWER:""")

# ── Module-level singletons (reused across warm invocations) ──────────────────

_collection = None
_embed      = None
_llm        = None


def _init():
    global _collection, _embed, _llm
    if _llm is not None:
        return

    api_key = os.environ["OPENAI_API_KEY"]

    # Load pre-computed embeddings (committed to repo — no API calls needed here)
    with open(DATA_FILE) as f:
        data = json.load(f)

    # Build in-memory ChromaDB — no disk writes, no /tmp, no read-only errors
    client = chromadb.Client()
    _collection = client.create_collection(COLLECTION_NAME)

    BATCH = 100
    for i in range(0, len(data), BATCH):
        batch = data[i : i + BATCH]
        _collection.add(
            ids=[d["id"] for d in batch],
            embeddings=[d["embedding"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[{"source": d["source"]} for d in batch],
        )

    _embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    _llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)


def _retrieve(question: str, k: int = 3) -> list:
    vec     = _embed.embed_query(question)
    results = _collection.query(query_embeddings=[vec], n_results=k)
    return [
        {"text": text, "source": meta.get("source", "")}
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def _answer(question: str, use_rag: bool) -> dict:
    _init()

    if use_rag:
        docs    = _retrieve(question)
        context = "\n\n".join(d["text"] for d in docs)
        chain   = RAG_PROMPT | _llm | StrOutputParser()
        answer  = chain.invoke({"context": context, "question": question})
        sources = sorted({d["source"] for d in docs})
    else:
        chain  = NO_RAG_PROMPT | _llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        sources = []

    return {"answer": answer, "sources": sources}


# ── Vercel handler ────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length   = int(self.headers.get("Content-Length", 0))
            body     = json.loads(self.rfile.read(length))
            question = body.get("question", "").strip()
            use_rag  = bool(body.get("use_rag", True))

            if not question:
                self._respond(400, {"error": "question is required"})
                return

            self._respond(200, _answer(question, use_rag))

        except KeyError as e:
            self._respond(500, {"error": f"Missing environment variable: {e}"})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _respond(self, status: int, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, *args):
        pass

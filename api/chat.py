"""
Vercel Python Serverless Function — POST /api/chat
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent.parent
_CHROMA_SRC     = BASE_DIR / "chroma_db"        # bundled (read-only on Vercel)
_CHROMA_TMP     = Path("/tmp/chroma_db")         # writable copy at runtime
COLLECTION_NAME = "technova_policies"


def _chroma_dir() -> str:
    """
    ChromaDB needs a writable directory even for reads (SQLite WAL files).
    On Vercel the bundle is read-only, so we copy it to /tmp on cold start.
    Subsequent warm invocations reuse the copy already in /tmp.
    """
    if not _CHROMA_TMP.exists():
        shutil.copytree(str(_CHROMA_SRC), str(_CHROMA_TMP))
    return str(_CHROMA_TMP)

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

# ── Module-level singletons (reused across warm invocations) ─────────────────

_retriever = None
_llm = None


def _init():
    global _retriever, _llm
    if _llm is not None:
        return

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    vectorstore = Chroma(
        persist_directory=_chroma_dir(),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    _llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _answer(question: str, use_rag: bool) -> dict:
    _init()

    if use_rag:
        docs = _retriever.invoke(question)
        context = _format_docs(docs)
        chain = RAG_PROMPT | _llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        sources = sorted({Path(d.metadata.get("source", "")).name for d in docs})
    else:
        chain = NO_RAG_PROMPT | _llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        sources = []

    return {"answer": answer, "sources": sources}


# ── Vercel handler ────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            question = body.get("question", "").strip()
            use_rag = bool(body.get("use_rag", True))

            if not question:
                self._respond(400, {"error": "question is required"})
                return

            result = _answer(question, use_rag)
            self._respond(200, result)

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
        pass  # suppress default access logs

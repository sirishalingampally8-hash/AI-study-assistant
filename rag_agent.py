"""
rag_agent.py - Core logic of the AI Study Assistant.

Pipeline:  notes -> chunks -> embeddings -> retrieve top-k -> LLM prompt
Features:  explain a concept, summarize a topic, generate MCQs.

The LLM is called through an OpenAI-compatible API, so you can use Groq,
Google Gemini, OpenAI, etc. just by changing values in the .env file.
"""

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class StudyAssistant:
    def __init__(self, chunk_words: int = 180, overlap_words: int = 40):
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.chunks: List[Dict] = []          # [{"text":..., "source":...}]
        self.embeddings: Optional[np.ndarray] = None
        self._embedder = None                 # lazy loaded
        self._client = None                   # lazy loaded

    # ------------------------------------------------------------------ #
    # Lazy loaders
    # ------------------------------------------------------------------ #
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("LLM_API_KEY")
            if not api_key:
                raise RuntimeError("LLM_API_KEY is missing. Add it to your .env file.")
            self._client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            )
        return self._client

    # ------------------------------------------------------------------ #
    # Knowledge base
    # ------------------------------------------------------------------ #
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if not words:
            return []
        step = max(1, self.chunk_words - self.overlap_words)
        chunks = []
        for start in range(0, len(words), step):
            piece = words[start:start + self.chunk_words]
            if piece:
                chunks.append(" ".join(piece))
            if start + self.chunk_words >= len(words):
                break
        return chunks

    def add_text(self, text: str, source: str = "notes") -> int:
        """Add raw text to the knowledge base and (re)build embeddings."""
        new_chunks = [{"text": c, "source": source} for c in self.chunk_text(text)]
        self.chunks.extend(new_chunks)
        self._build_index()
        return len(new_chunks)

    def add_file(self, path: str) -> int:
        """Add a .txt or .pdf file."""
        name = os.path.basename(path)
        if path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        return self.add_text(text, source=name)

    def _build_index(self):
        if not self.chunks:
            self.embeddings = None
            return
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def clear(self):
        self.chunks, self.embeddings = [], None

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if self.embeddings is None or not self.chunks:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def _context(self, query: str, k: int) -> str:
        hits = self.retrieve(query, k)
        if not hits:
            return ""
        return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

    # ------------------------------------------------------------------ #
    # LLM call
    # ------------------------------------------------------------------ #
    def _chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        response = self.client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # Feature 1: Explain
    # ------------------------------------------------------------------ #
    def explain(self, question: str, level: str = "Beginner") -> str:
        context = self._context(question, k=4)
        system = (
            "You are a friendly study tutor. Explain concepts clearly with a simple "
            f"example. Adjust the depth for a {level} student. Prefer the provided "
            "notes; if they do not cover the question, say so briefly and then "
            "answer from general knowledge."
        )
        user = f"NOTES:\n{context or '(no notes uploaded)'}\n\nQUESTION: {question}"
        return self._chat(system, user)

    # ------------------------------------------------------------------ #
    # Feature 2: Summarize
    # ------------------------------------------------------------------ #
    def summarize(self, topic: str = "", style: str = "Bullet points") -> str:
        if topic.strip():
            context = self._context(topic, k=6)
        else:  # summarize everything (capped to keep the prompt small)
            context = "\n\n".join(c["text"] for c in self.chunks)[:12000]
        if not context:
            return "No notes available. Please upload or add notes first."
        system = (
            "You are a study assistant that writes accurate, concise summaries "
            "using ONLY the notes provided."
        )
        user = (
            f"Summarize the notes{' on: ' + topic if topic.strip() else ''}.\n"
            f"Format: {style}. Highlight key terms in **bold**.\n\nNOTES:\n{context}"
        )
        return self._chat(system, user)

    # ------------------------------------------------------------------ #
    # Feature 3: MCQs
    # ------------------------------------------------------------------ #
    def generate_mcqs(self, topic: str = "", n: int = 5, difficulty: str = "Medium") -> List[Dict]:
        if topic.strip():
            context = self._context(topic, k=6)
        else:
            context = "\n\n".join(c["text"] for c in self.chunks)[:12000]
        if not context:
            raise ValueError("No notes available. Please upload or add notes first.")

        system = "You are an exam question writer. Output ONLY valid JSON, no extra text."
        user = f"""Create {n} {difficulty}-difficulty multiple choice questions from the notes below{' about: ' + topic if topic.strip() else ''}.

Return a JSON array. Each item must look exactly like:
{{"question": "...", "options": ["A text", "B text", "C text", "D text"], "answer": 0, "explanation": "..."}}

Rules:
- exactly 4 options per question, only one correct
- "answer" is the index (0-3) of the correct option
- base questions on the notes only

NOTES:
{context}"""
        raw = self._chat(system, user, temperature=0.4)
        return self.parse_mcqs(raw)

    @staticmethod
    def parse_mcqs(raw: str) -> List[Dict]:
        """Extract and validate the MCQ JSON returned by the LLM."""
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("Model did not return a JSON list of questions.")
        data = json.loads(raw[start:end + 1])

        cleaned = []
        for item in data:
            options = item.get("options", [])
            if not item.get("question") or len(options) != 4:
                continue
            ans = item.get("answer", 0)
            if isinstance(ans, str):
                ans = ans.strip().upper()
                ans = "ABCD".index(ans[0]) if ans[:1] in "ABCD" and not ans.isdigit() else int(ans)
            if not 0 <= int(ans) <= 3:
                continue
            cleaned.append({
                "question": item["question"],
                "options": options,
                "answer": int(ans),
                "explanation": item.get("explanation", ""),
            })
        if not cleaned:
            raise ValueError("Could not parse any valid questions. Try again.")
        return cleaned

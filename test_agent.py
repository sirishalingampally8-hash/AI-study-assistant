"""
test_agent.py - Quick tests for the AI Study Assistant.

Run:  pytest test_agent.py -v      (or)      python test_agent.py

The first group of tests works offline (no API key needed).
The live tests run only if LLM_API_KEY is set in your .env.
"""

import os

import pytest
from dotenv import load_dotenv

from rag_agent import StudyAssistant

load_dotenv()
HAS_KEY = bool(os.getenv("LLM_API_KEY"))


@pytest.fixture(scope="module")
def assistant():
    a = StudyAssistant()
    a.add_file("study_notes.txt")
    return a


# ---------------------- offline tests ---------------------------------- #
def test_chunking_overlap():
    a = StudyAssistant(chunk_words=10, overlap_words=2)
    chunks = a.chunk_text(" ".join(f"w{i}" for i in range(30)))
    assert len(chunks) >= 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_knowledge_base_built(assistant):
    assert len(assistant.chunks) > 0
    assert assistant.embeddings.shape[0] == len(assistant.chunks)


def test_retrieval_finds_relevant_chunk(assistant):
    hits = assistant.retrieve("What are the four conditions of deadlock?", k=2)
    assert any("deadlock" in h["text"].lower() for h in hits)


def test_retrieval_on_dbms(assistant):
    hits = assistant.retrieve("ACID properties of a transaction", k=2)
    assert any("atomicity" in h["text"].lower() for h in hits)


def test_parse_mcqs_handles_code_fences():
    raw = """```json
    [{"question": "Q?", "options": ["a","b","c","d"], "answer": "B", "explanation": "x"}]
    ```"""
    mcqs = StudyAssistant.parse_mcqs(raw)
    assert mcqs[0]["answer"] == 1


def test_parse_mcqs_rejects_garbage():
    with pytest.raises(ValueError):
        StudyAssistant.parse_mcqs("sorry, I cannot do that")


# ---------------------- live tests (need API key) ---------------------- #
@pytest.mark.skipif(not HAS_KEY, reason="LLM_API_KEY not set")
def test_explain(assistant):
    out = assistant.explain("What is a deadlock?")
    assert len(out) > 50


@pytest.mark.skipif(not HAS_KEY, reason="LLM_API_KEY not set")
def test_summarize(assistant):
    assert len(assistant.summarize("Normalization")) > 30


@pytest.mark.skipif(not HAS_KEY, reason="LLM_API_KEY not set")
def test_generate_mcqs(assistant):
    mcqs = assistant.generate_mcqs("Operating systems", n=3)
    assert 1 <= len(mcqs) <= 3
    assert all(len(q["options"]) == 4 for q in mcqs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

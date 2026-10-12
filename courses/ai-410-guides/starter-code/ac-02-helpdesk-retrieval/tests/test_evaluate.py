"""
Provided tests — your code must pass all of these. Run:  pytest

Don't edit this file. Your own tests go in test_my_rules.py.
"""

import json
import pathlib

import pytest

import evaluate
from embedders import LocalEmbedder
from vendor_pipeline import build_index, load_articles, search, vendor_chunks

EVAL = json.loads((pathlib.Path(__file__).parent.parent / "eval_set.json").read_text())
EMB = LocalEmbedder()


def fake_search(results_by_question):
    """A search_fn that returns canned results, so tests don't depend on chunking."""
    def search_fn(question, k):
        return results_by_question.get(question, [])[:k]
    return search_fn


MINI = [
    {"id": "a", "question": "qa", "source": "doc1", "answer": "Blue Door"},
    {"id": "b", "question": "qb", "source": "doc2", "answer": "42 days"},
    {"id": "c", "question": "qc", "source": None, "answer": None},
]
CANNED = fake_search({
    "qa": [{"source": "doc1", "text": "Go through the blue door.", "score": 0.9}],
    "qb": [{"source": "doc1", "text": "It takes 42 days.", "score": 0.8},
           {"source": "doc2", "text": "It takes 42 days.", "score": 0.7}],
    "qc": [{"source": "doc1", "text": "nothing", "score": 0.1}],
})


# --- hit_rate ----------------------------------------------------------

def test_hit_needs_right_source_and_answer_ignoring_case():
    # qa: hit. qb at k=1: right text, wrong source -> miss. qc: not counted.
    assert evaluate.hit_rate(MINI, CANNED, 1) == {"hit_rate": 0.5, "misses": ["b"]}


def test_hit_rate_uses_k():
    assert evaluate.hit_rate(MINI, CANNED, 2) == {"hit_rate": 1.0, "misses": []}


def test_vendor_baseline_on_the_real_questions():
    index = build_index(vendor_chunks(), EMB)
    result = evaluate.hit_rate(EVAL, lambda q, k: search(index, q, EMB, k), 1)
    assert result["hit_rate"] == 0.5625


# --- chunk_article -----------------------------------------------------

@pytest.fixture(scope="module")
def your_chunks():
    return [c for aid, text in load_articles().items() for c in evaluate.chunk_article(aid, text)]


def test_chunks_are_labelled_non_empty_and_small_enough(your_chunks):
    articles = load_articles()
    assert your_chunks, "chunk_article returned nothing"
    for c in your_chunks:
        assert c["source"] in articles
        assert c["text"].strip(), "empty chunk"
        assert len(c["text"]) <= evaluate.MAX_CHUNK_CHARS, f"{len(c['text'])}-character chunk from {c['source']}"


def test_no_answer_is_lost(your_chunks):
    for q in EVAL:
        if q["source"]:
            texts = [c["text"].lower() for c in your_chunks if c["source"] == q["source"]]
            assert any(q["answer"].lower() in t for t in texts), (
                f"{q['id']}: no chunk of {q['source']} still contains {q['answer']!r}"
            )


def test_your_chunks_beat_the_vendor_at_k1(your_chunks):
    index = build_index(your_chunks, EMB)
    yours = evaluate.hit_rate(EVAL, lambda q, k: search(index, q, EMB, k), 1)["hit_rate"]
    assert yours > 0.5625, f"your hit@1 is {yours}; the vendor's is 0.5625"


# --- answer_or_abstain -------------------------------------------------

def test_abstains_below_threshold():
    assert evaluate.answer_or_abstain("qc", CANNED, 0.5) == {"abstain": True, "top_score": 0.1}


def test_answers_at_or_above_threshold():
    assert evaluate.answer_or_abstain("qa", CANNED, 0.9) == {
        "abstain": False, "source": "doc1", "context": "Go through the blue door.", "top_score": 0.9,
    }


def test_abstains_when_nothing_comes_back():
    assert evaluate.answer_or_abstain("unknown", CANNED, 0.1) == {"abstain": True, "top_score": 0.0}


def test_defaults_to_your_min_score():
    assert 0 < evaluate.MIN_SCORE < 1, "set MIN_SCORE to your chosen threshold"
    d = evaluate.answer_or_abstain("qa", fake_search({"qa": [{"source": "doc1", "text": "t", "score": evaluate.MIN_SCORE}]}))
    assert d["abstain"] is False

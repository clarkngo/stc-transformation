"""
Reference solution for Application Challenge 2 (faculty only).

Chunking: one chunk per "##" section, body only. With the offline
embedder this scores hit@1 0.81 against the vendor's 0.56. Adding the
title or heading to each chunk scores LOWER offline (extra words dilute
a word-count vector), a good thing for a student to notice and explain.

MIN_SCORE = 0.25 is ONE defensible choice. In run_eval.py's sweep it
answers 10 correctly, answers 3 wrongly (q05 and q16 from the wrong
section, and out-of-scope x04 at 0.28, the closest call), and declines
4 answerable ones (q02, q07, q15, and q09, which was a miss anyway).
0.30 cuts wrong answers to 1 but loses 6; the reasoning is what's graded.
"""

import re

MAX_CHUNK_CHARS = 800
MIN_SCORE = 0.25


def hit_rate(eval_set, search_fn, k):
    counted = [q for q in eval_set if q["source"] is not None]
    misses = []
    for q in counted:
        results = search_fn(q["question"], k)
        if not any(r["source"] == q["source"] and q["answer"].lower() in r["text"].lower() for r in results):
            misses.append(q["id"])
    return {"hit_rate": round((len(counted) - len(misses)) / len(counted), 4), "misses": misses}


def chunk_article(article_id, text):
    chunks = []
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        _heading, _, body = section.partition("\n")
        body = body.strip()
        while body:  # sections are short here, but never exceed the cap
            chunks.append({"source": article_id, "text": body[:MAX_CHUNK_CHARS]})
            body = body[MAX_CHUNK_CHARS:]
    return chunks


def answer_or_abstain(question, search_fn, min_score=None):
    min_score = MIN_SCORE if min_score is None else min_score
    results = search_fn(question, 1)
    if not results or results[0]["score"] < min_score:
        return {"abstain": True, "top_score": results[0]["score"] if results else 0.0}
    top = results[0]
    return {"abstain": False, "source": top["source"], "context": top["text"], "top_score": top["score"]}

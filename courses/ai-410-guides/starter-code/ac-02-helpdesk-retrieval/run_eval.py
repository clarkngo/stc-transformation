"""
Measure the vendor's chunking against yours, on eval_set.json.
Paste the output into REPORT.md.

    python run_eval.py            # offline, LocalEmbedder (same numbers for everyone)
    python run_eval.py --gemini   # optional: real Gemini embeddings (needs .env key)
"""

import json
import pathlib
import sys

import evaluate
from embedders import GeminiEmbedder, LocalEmbedder
from vendor_pipeline import build_index, load_articles, search, vendor_chunks

EVAL = json.loads((pathlib.Path(__file__).parent / "eval_set.json").read_text())


def your_chunks() -> list[dict]:
    return [c for aid, text in load_articles().items() for c in evaluate.chunk_article(aid, text)]


def main() -> None:
    gemini = "--gemini" in sys.argv
    if gemini:
        # The vendor embeds documents with the QUERY task type — reproduce that.
        vendor_emb, your_emb = GeminiEmbedder("RETRIEVAL_QUERY"), GeminiEmbedder("RETRIEVAL_DOCUMENT")
    else:
        vendor_emb = your_emb = LocalEmbedder()
    print(f"Embedder: {your_emb.name}\n")

    runs = {}
    for label, chunks, emb in [("vendor", vendor_chunks(), vendor_emb), ("yours", your_chunks(), your_emb)]:
        index = build_index(chunks, emb)
        runs[label] = lambda q, k, index=index, emb=emb: search(index, q, emb, k)
        print(f"{label:7} {len(chunks):3} chunks", end="")
        for k in (1, 3):
            r = evaluate.hit_rate(EVAL, runs[label], k)
            print(f" | hit@{k} {r['hit_rate']:.2f}", end="")
        print()

    print("\nYour misses at k=1 (question -> what came back instead):")
    for qid in evaluate.hit_rate(EVAL, runs["yours"], 1)["misses"]:
        q = next(q for q in EVAL if q["id"] == qid)
        top = runs["yours"](q["question"], 1)[0]
        print(f"  {qid} {q['question']}\n      expected {q['source']} -> got {top['source']} "
              f"(score {top['score']:.2f}): {top['text'][:70]!r}")

    print("\nThreshold sweep with your chunks (top result only):")
    print("  min_score | answered right | answered WRONG | abstained on answerable")
    thresholds = sorted({round(t / 100, 2) for t in range(10, 45, 5)} | {round(evaluate.MIN_SCORE, 2)})
    for t in thresholds:
        right = wrong = lost = 0
        for q in EVAL:
            d = evaluate.answer_or_abstain(q["question"], runs["yours"], t)
            correct = (q["source"] is not None and not d["abstain"] and d["source"] == q["source"]
                       and q["answer"].lower() in d["context"].lower())
            if d["abstain"]:
                lost += q["source"] is not None
            elif correct:
                right += 1
            else:
                wrong += 1
        mark = "  <- your MIN_SCORE" if t == round(evaluate.MIN_SCORE, 2) else ""
        print(f"     {t:.2f}  |      {right:2}        |      {wrong:2}        |        {lost:2}{mark}")


if __name__ == "__main__":
    main()

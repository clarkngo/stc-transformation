"""
YOUR tests for your MIN_SCORE decision, on the REAL questions and YOUR
chunks. Write at least two:
  - one question you decided the bot must still answer, and
  - one it must decline (for example an x-question from eval_set.json),
and, ideally, the case closest to your threshold that worries you most.

Build a real search_fn like this:

    from embedders import LocalEmbedder
    from vendor_pipeline import build_index, load_articles, search
    import evaluate

    EMB = LocalEmbedder()
    INDEX = build_index(
        [c for aid, t in load_articles().items() for c in evaluate.chunk_article(aid, t)], EMB)
    search_fn = lambda q, k: search(INDEX, q, EMB, k)
"""


def test_replace_me():
    ...

"""
YOUR WORK GOES HERE — measure the help desk's retrieval, improve it,
and decide when the bot should say "I don't know".

Three functions and one number. The tests in tests/test_evaluate.py
check them; `python run_eval.py` shows what they do to the real
questions in eval_set.json.

A `search_fn` is any function search_fn(question, k) that returns the
k best chunks as [{"source": ..., "text": ..., "score": ...}, ...],
best first (vendor_pipeline.search, wrapped, is one).
"""

MAX_CHUNK_CHARS = 800  # the chat model's prompt has room for one chunk this size per result

# YOUR DECISION (see the challenge spec): below what top score should
# the bot stop answering and send the student to a person instead?
MIN_SCORE = 0.0


def hit_rate(eval_set: list[dict], search_fn, k: int) -> dict:
    """How often the top-k results contain the answer.

    Count only questions whose "source" is not None. A question is a
    HIT when at least one of its top-k results has the expected source
    AND contains the expected "answer" phrase (ignore upper/lower case).

    Returns {"hit_rate": hits / questions counted, rounded to 4 places,
             "misses": [ids of the questions that were not hits, in order]}.
    """
    raise NotImplementedError


def chunk_article(article_id: str, text: str) -> list[dict]:
    """Your chunking strategy, for one article.

    Returns [{"source": article_id, "text": ...}, ...]. Every chunk must
    be non-empty and at most MAX_CHUNK_CHARS long, and no answer may be
    lost (every fact in the article ends up in some chunk). Beyond that
    it's your design: by heading, by paragraph, with overlap, with the
    title added — measure, then choose.
    """
    raise NotImplementedError


def answer_or_abstain(question: str, search_fn, min_score: float = None) -> dict:
    """Decide whether the bot should answer from the top result.

    min_score defaults to MIN_SCORE. Look at the single best result:
      - none, or its score is below min_score ->
            {"abstain": True, "top_score": score or 0.0}
      - otherwise ->
            {"abstain": False, "source": ..., "context": its text, "top_score": score}
    """
    raise NotImplementedError

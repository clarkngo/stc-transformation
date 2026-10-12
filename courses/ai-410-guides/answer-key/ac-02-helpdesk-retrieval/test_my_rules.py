"""Reference tests for the answer key's MIN_SCORE = 0.25."""

import evaluate
from embedders import LocalEmbedder
from vendor_pipeline import build_index, load_articles, search

EMB = LocalEmbedder()
INDEX = build_index([c for aid, t in load_articles().items() for c in evaluate.chunk_article(aid, t)], EMB)


def search_fn(q, k):
    return search(INDEX, q, EMB, k)


def test_still_answers_the_print_credit_question():
    d = evaluate.answer_or_abstain("How many free pages of printing do I get each term?", search_fn)
    assert d["abstain"] is False and "300 free pages" in d["context"]


def test_declines_parking():
    assert evaluate.answer_or_abstain("Can I park my car overnight in the student lot?", search_fn)["abstain"] is True


def test_known_risk_wifi_printer_question_still_slips_through():
    # x04 scores 0.28: above the threshold, so the bot answers from the
    # passwords article. Documented on purpose; see DECISIONS.md.
    d = evaluate.answer_or_abstain("How do I change my password on the school's Wi-Fi printer?", search_fn)
    assert d["abstain"] is False and d["source"] == "accounts-passwords"

# Decisions — Application Challenge 2

Keep it to about one page (300–500 words). Graded on the quality of your reasoning, not on matching one "right" answer. Section 1 is graded as **Evaluation** (30%), section 2 as **Analysis** (20%).

## 1. When should the bot say "I don't know"? — Evaluation

The help desk lead: "A wrong answer sends a student to the wrong office, or makes them miss a deadline. But 'I don't know' sends them to the ticket queue, and that's a two-day wait right now." Nobody set a threshold.

- **My MIN_SCORE:** the number, and the row of the sweep in `run_eval.py` it corresponds to.
- **Why:** which mistake is worse here, a wrong answer or a lost one, and for whom? Name at least one other threshold you rejected and what it would have cost.
- **My tests:** which tests in `tests/test_my_rules.py` prove the rule, and what edge does each cover?

## 2. Why the misses happen — Analysis

From `run_eval.py`, take the vendor's misses and your remaining misses. Sort them by cause: **chunking** (the answer was cut off or diluted), **vocabulary** (the student's words don't match the article's words), or **something else**. For each cause, name one question it explains and what would fix it. Which of your fixes is chunking actually able to make, and which would need something else (real embeddings, synonyms, more articles)?

## 3. AI use

Which parts did an AI assistant write or suggest? What did you change, and what did it get wrong or miss?

# Decisions — Application Challenge 4

Keep it to about one page (300–500 words). Graded on the quality of your reasoning, not on matching one "right" answer. Section 1 is graded as **Evaluation** (30%), section 2 as **Analysis** (20%).

## 1. What pages the on-call engineer — Evaluation

The operations lead: "Page us if substitutions go wrong. But if the pager goes off for nothing, people start ignoring it, and then it's useless." Nobody said what "go wrong" means.

- **My rule:** what makes `should_page` return True, and which orders it pages for in `run_day.py`.
- **Why:** what makes those worth waking someone at 3 a.m. (how rare, how harmful, can someone act right now)? Name at least one condition you put on the dashboard instead, and why it doesn't page.
- **My tests:** which tests in `tests/test_my_rules.py` prove the rule, and what edge does each cover?

## 2. What the trace lets you see — Analysis

Break the feature into its four steps. For each problem in the `run_day.py` report (a quality 0.0, a quality 0.5, the slow call), say which span shows it and what you'd have seen with only one overall "order processed" log line instead. Then name one thing the feature gets wrong that your trace still can't show you.

## 3. AI use

Which parts did an AI assistant write or suggest? What did you change, and what did it get wrong or miss?

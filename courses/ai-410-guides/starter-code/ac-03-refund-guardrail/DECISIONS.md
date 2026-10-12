# Decisions — Application Challenge 3

Keep it to about one page (300–500 words). Graded on the quality of your reasoning, not on matching one "right" answer. Section 1 is graded as **Evaluation** (30%), section 2 as **Analysis** (20%).

## 1. What the bot may refund on its own — Evaluation

The owner: "Handle refunds so I don't have to. But never let the bot refund more than $500 on its own." Everything at or under $500 is your call.

- **My rules:** your `AUTO_REFUND_LIMIT`, and any other rule that sends a refund to the review queue (a reason, a repeat refund, an account type, ...).
- **Why:** who do your rules protect, and what do they cost (the owner's time, the customer's wait)? Name at least one alternative you rejected and why.
- **My tests:** which tests in `tests/test_my_rules.py` prove the rules, and what edge does each cover?

## 2. Where each check belongs — Analysis

Take the bad calls in `recorded_calls.json`. For each one, say which layer stops it in your code: the **schema** (RefundArgs), a **lookup** (the order data), or a **person** (the review queue), and why it can't be stopped at an earlier layer. Then name one way a call could still pass every check you wrote and still be a refund the owner wouldn't want.

## 3. AI use

Which parts did an AI assistant write or suggest? What did you change, and what did it get wrong or miss?

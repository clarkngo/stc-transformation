# Decisions — Application Challenge 1

Keep it to about one page (300–500 words). Graded on the quality of your reasoning, not on matching one "right" answer. Section 1 is graded as **Evaluation** (30%), section 2 as **Analysis** (20%).

## 1. Late cancellations (the open decision) — Evaluation

The front desk asked for: "Patients should be able to cancel through the chat." The clinic charges a $40 fee for cancelling less than 24 hours ahead. Nobody said what the agent should do inside that window.

- **My rule:** what does `cancel_appointment` do when the appointment is less than 24 hours away?
- **Why:** who does your rule protect, and what does it cost? Name at least one alternative you rejected and why.
- **My tests:** which tests in `tests/test_my_rules.py` prove the rule, and what edge does each one cover?

## 2. The rest of the wish list — Analysis

The clinic manager also asked for these. You don't build them. For each, decide: **expose it to the model**, **expose it with limits**, or **don't expose it**. If you'd expose it, write its schema (name, description, parameters). "Don't expose it" with a good reason is a strong answer.

For each tool, break down what it would actually let the model do: what data it could see, and what actions it could take.

- **`view_patient_record`** — "so the agent can answer questions about a patient's history."
- **`send_sms`** — "so the agent can text patients reminders."

## 3. AI use

Which parts did an AI assistant write or suggest? What did you change, and what did it get wrong or miss?

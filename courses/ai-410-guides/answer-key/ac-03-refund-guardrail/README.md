# Application Challenge 3 — answer key (faculty)

`refunds.py` and `test_my_rules.py` are one complete, passing submission. Its routing rules are one defensible choice, not the required one; grade the student's rules on the reasoning in `DECISIONS.md`, using the faculty notes in the guide.

To check it against the provided tests:

```bash
cd ../../starter-code/ac-03-refund-guardrail
cp ../../answer-key/ac-03-refund-guardrail/refunds.py .
cp ../../answer-key/ac-03-refund-guardrail/test_my_rules.py tests/
pytest && python run_cases.py
git checkout refunds.py tests/test_my_rules.py   # put the student version back
```

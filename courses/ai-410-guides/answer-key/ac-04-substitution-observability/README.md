# Application Challenge 4 — answer key (faculty)

`instrumented.py` and `test_my_rules.py` are one complete, passing submission. Its paging rule is one defensible choice, not the required one; grade the student's rule on the reasoning in `DECISIONS.md`, using the faculty notes in the guide.

To check it against the provided tests:

```bash
cd ../../starter-code/ac-04-substitution-observability
cp ../../answer-key/ac-04-substitution-observability/instrumented.py .
cp ../../answer-key/ac-04-substitution-observability/test_my_rules.py tests/
pytest && python run_day.py
git checkout instrumented.py tests/test_my_rules.py   # put the student version back
```

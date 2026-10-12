# Application Challenge 2 — answer key (faculty)

`evaluate.py` and `test_my_rules.py` are one complete, passing submission. Its `MIN_SCORE` is one defensible choice, not the required one; grade the student's threshold on the reasoning in `DECISIONS.md`, using the faculty notes in the guide.

To check it against the provided tests:

```bash
cd ../../starter-code/ac-02-helpdesk-retrieval
cp ../../answer-key/ac-02-helpdesk-retrieval/evaluate.py .
cp ../../answer-key/ac-02-helpdesk-retrieval/test_my_rules.py tests/
pytest && python run_eval.py
git checkout evaluate.py tests/test_my_rules.py   # put the student version back
```

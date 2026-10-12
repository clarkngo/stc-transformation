# Application Challenge 1 — answer key (faculty)

`tools.py` and `test_my_rules.py` are one complete, passing submission. The late-cancellation rule in it (refuse inside 24 hours, send the patient to the front desk) is one defensible choice, not the required one. Grade the student's rule on the reasoning in their `DECISIONS.md`, using the faculty notes in the guide.

To check it against the provided tests:

```bash
cd ../../starter-code/ac-01-clinic-booking
cp ../../answer-key/ac-01-clinic-booking/tools.py .
cp ../../answer-key/ac-01-clinic-booking/test_my_rules.py tests/
pytest
git checkout tools.py tests/test_my_rules.py   # put the student version back
```

# Reference: Analyze stage

A strong Analyze write-up picks ONE mechanism and explains it precisely — not the whole file. The most natural target in this HOS is the function-call detection in `backend/agent.py`'s `run_agent()`. Reference walkthrough, for comparing depth against a student's submission:

```python
function_calls = [step for step in interaction.steps if step.type == "function_call"]

if not function_calls:
    return interaction.output_text
```

A submission that's actually understood this, not just pattern-matched it, should be able to explain:

1. **What `interaction.steps` is.** Every call to `client.interactions.create()` returns an `interaction` object, and `steps` is a list describing everything the model did to produce its response — it might be just one text step, or it might include one or more `function_call` steps if the model decided to use a tool.

2. **Why the check is a list comprehension, not a single `if`.** The model can request more than one tool call in the same turn (e.g. it might call `calculate` and `word_count` in response to a single message that needs both). The code has to handle zero, one, or several `function_call` steps in one pass — that's why `function_calls` is built as a list before anything is decided.

3. **Why `if not function_calls` means "the model is done."** If there are no `function_call` steps, the model chose to answer directly instead of asking to run something — `interaction.output_text` is that direct answer, and the loop returns immediately instead of continuing.

4. **What happens on the other branch.** Each `function_call` in the list gets matched to a real Python function via `TOOL_FUNCTIONS.get(call.name)`, run locally, and its result packaged as a `function_result` — sent back to the model via `previous_interaction_id` so it can incorporate the result and decide what to do next (answer, or call another tool).

A write-up that says "it checks if there's a function call and runs it" hasn't actually analyzed this — it's restated the code in English. The bar is explaining *why* it's shaped this way (a list, not a boolean; a loop, not a single pass) in terms of what the model is actually allowed to do.

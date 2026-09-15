# Reference: Evaluate stage

What a strong Evaluate write-up should surface, for comparing against a student's submission. This isn't a checklist to hand to students verbatim — it's what a genuine test pass tends to turn up.

## Things a typical AI-scaffolded first pass gets right
- The backend holds `GEMINI_API_KEY`, never the frontend — a reasonable default even without being asked.
- `/chat` returns JSON with a `reply` field, matching what the frontend expects.
- The agent answers a plain question ("what's the capital of France?") without calling any tool.

## Things worth specifically testing, and what commonly goes wrong
- **A message that clearly needs the tool** (e.g. "what's 342 times 87?") — does the agent actually call `calculate`, or does it try to compute the answer itself in text and get it wrong? A scaffolded loop that doesn't check `step.type == "function_call"` correctly will silently skip the tool and just guess.
- **A message that's ambiguous** ("what's 342 times 87, roughly?") — some scaffolds call the tool anyway, some don't. Neither is "wrong," but the student should notice and say which happened.
- **An expression the calculator can't handle** (e.g. `"342 * "`, incomplete) — a naive `eval()`-based first pass will crash the whole request instead of returning a graceful error. This is the single most common gap in a fast AI-scaffolded `calculate` tool, and worth calling out explicitly if the student's build has it.
- **`MAX_TURNS` / loop termination** — ask what happens if the model calls a tool, gets a result, and calls a *different* tool in response, repeatedly. Does the loop actually stop after 5 turns, or does it hang? Most scaffolds get the happy path right and never test this.
- **Empty/whitespace-only message** — does the backend 500, or does it get a sensible (if unhelpful) reply?

A submission that only tests the happy path hasn't actually evaluated anything — it's re-confirmed the demo works. Look for evidence the student tried to break it.

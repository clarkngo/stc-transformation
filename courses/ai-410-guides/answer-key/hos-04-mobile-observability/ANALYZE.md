# Reference: Analyze stage

Two natural targets — a strong submission goes deep on one.

## Option A: why push notifications need a device-specific token, not a user ID

`devices.py` stores `{device_id: push_token}`, and `jobs.py` looks up the token by `device_id` at the moment a job finishes — it never stores or reasons about "the user." A submission that's understood this should be able to explain: a push token identifies one specific installation on one specific device, issued by Apple/Google's push infrastructure and handed to Expo — it's not a stable identifier for a *person*. If the same person opens the app on two phones, that's two tokens; reinstalling the app can issue a new token for the same device. Storing by `device_id` (not "user") is what makes "notify whoever's phone requested this job" actually correct — a user-keyed store would either miss devices or need to fan a notification out to every device that user has ever registered, which isn't what actually happened here (one specific phone asked for one specific job).

## Option B: why retrieval and generation are traced as separate spans

```python
@observe()
def retrieve(query: str, k: int = 5) -> list[str]: ...

@observe()
def run_agent(user_message: str) -> str:
    chunks = retrieve(user_message, k=5)
    ...
```

Both functions carry their own `@observe()`, and `run_agent` calls `retrieve` from inside its own traced scope — Langfuse nests the resulting spans automatically based on the call stack, without either function needing to know about the other's tracing. A submission that's understood this should be able to explain *why* that nesting matters: a single flat trace around the whole request would tell you "this call took 2.3 seconds" and nothing else; two nested spans let you see the split — e.g. 200ms for the embedding + vector search versus 2.1s for the Gemini call — which is the difference between "something's slow" and "the model call is the bottleneck, not retrieval." That's the actual point of tracing at this granularity: turning one number into a breakdown you can act on.

A write-up that says "it traces the function" for either option hasn't analyzed anything — the bar is explaining what the design choice makes visible that a simpler version wouldn't.

# Reference: Evaluate stage

## Things a typical AI-scaffolded first pass gets right
- `POST /register-device` stores a token keyed by `device_id`, not by user — reasonable given there's no auth system in this course.
- `@observe()` on `run_agent` and `retrieve` produces two separate spans per request, not one merged one.

## Things worth specifically testing
- **A push notification actually arrives on the registered device**, not just that the API call succeeds. `send_push_notification()`'s `response.raise_for_status()` only catches HTTP-level failures — Expo's push API can return **200 OK with an error in the response body** (e.g. `DeviceNotRegistered` for a stale or fake token). A submission that only checks the HTTP status hasn't actually verified delivery.
- **A trace in the Langfuse dashboard for a real request** — does it show retrieval and generation as separate spans, or one flat call? Can you find the token/cost numbers and latency for each?
- **The faithfulness score on an answer you know is wrong** — deliberately ask something the docs don't cover and see what the model does; if it hallucinates instead of saying "I don't know," does the logged faithfulness score reflect that (low), or does the heuristic miss it?
- **Whether tracing/scoring measurably slows down `/chat`** — time a request with and without `LANGFUSE_*` keys set. A heavy synchronous tracing setup can add noticeable latency; Langfuse's SDK is supposed to batch/flush asynchronously, but it's worth actually timing rather than assuming.

Note: `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` unset doesn't break anything — the SDK disables itself with a log line and the app keeps working. That's a legitimate thing for a student to discover and report, not a bug to "fix" by making tracing mandatory.

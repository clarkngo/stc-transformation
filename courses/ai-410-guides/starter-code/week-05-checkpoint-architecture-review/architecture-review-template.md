# Architecture & Bottleneck Review

**Name:**
**Date:**

## 1. Request path diagram

Draw (or paste an image of) the request path exactly as it exists in your code today:

```
User message → frontend → backend → embed query → vector search → inject context → call Gemini → (maybe) tool call → response → frontend
```

Redraw the arrows above to match what your app *actually* does — add or remove steps as needed.

## 2. Interrogate every arrow

For each step, answer both questions in one sentence each.

| Step | What if it's slow? | What if it returns something malformed/unexpected? |
|---|---|---|
| e.g. Vector search | | |
| | | |
| | | |
| | | |
| | | |

## 3. Named failure points (at least 3)

Be specific — tie each one to an exact place in *your* code, not a generic risk.

### Failure point 1
- **Where:** (file/function)
- **What a user would see:**
- **Proposed mitigation:**

### Failure point 2
- **Where:**
- **What a user would see:**
- **Proposed mitigation:**

### Failure point 3
- **Where:**
- **What a user would see:**
- **Proposed mitigation:**

## 4. Instructor review notes

(filled in live during your Week 5 review)

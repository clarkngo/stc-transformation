# HOS 3 — Reference failure log

What a strong Stage 2 submission finds, from the traces alone. Real model runs vary, so a student's
"first step where it shows" may differ by a step. Grade the reasoning, not an exact match.

| Scenario | Failure type | First step where it shows | Symptom | Root cause (confirmed in Stage 3) |
|---|---|---|---|---|
| s1-order-lookup | Tool-call error | The first `get_order` call, args `{"order_id": "#1004"}` | Run ends in an error: `ValueError: invalid literal for int() with base 10: '#1004'`. The customer gets no reply. | `get_order`'s docstring tells the model to pass the ID "exactly as the customer wrote it, including any '#'", but the code does `int(order_id)`. The schema and the code disagree. |
| s2-refund | Invented action (hallucinated tool) | The `issue_refund` call; its result is an error saying no such tool exists | The agent often tells the customer the refund **has been processed**. Nothing was processed. | The instruction promises an `issue_refund` tool that was never registered. ADK sends back an error, and the model covers for it. This is the most dangerous failure: polite, confident, and false. |
| s3-where-is-it | Logic loop | The second identical `check_shipment(1002)` call | The same call repeats until the step budget ends the run (`LlmCallsLimitExceededError`), with no reply to the customer. | The tool always says "check again for the latest status", and the instruction says "always confirm the latest tracking status". The model obeys both, forever. |
| s4-monthly-total | Wrong data (silent truncation) | The `list_orders` result: only 5 orders, IDs 1001–1005 | The agent confidently reports **5 orders, $1,622.79**. The truth is **8 orders, $2,192.49**. | The query ends in a leftover `LIMIT 5`, and nothing in the result says rows are missing. Catching it means noticing the order IDs stop at 1005, not reading an error. |
| s5-price-quote | Wrong tool | The agent calls `get_price` instead of `get_product` | The customer is quoted **$153.30**. The retail price is **$219.00**. | `get_price` returns the store's wholesale cost (retail × 0.7) under the vague key `price`, and its description ("Get the price of a product") matches the question better than `get_product`'s does. |
| s6-control | None | — | Correct list of safety products. | Control scenario: proves the agent and tools work when nothing is planted. |

## Fixes in this answer key

| # | Fix | Where |
|---|---|---|
| 1 | Parse `#1004` / `ORD-1004` / `1004`; return an error dict for input with no number | `tools.py` `_parse_order_id` |
| 2 | Remove the `issue_refund` promise; refunds become a `create_support_ticket` handoff to a human | `tools.py`, instruction |
| 3 | `check_shipment` returns a definite status; instruction says to call it once | `tools.py`, instruction |
| 4 | `list_orders` returns every row, plus `count` and `combined_value` computed in code | `tools.py` |
| 5 | Remove `get_price` (staff-only data) from the customer-facing agent; `get_product` says `retail_price` | `tools.py`, instruction |
| Generic | `error_as_observation` (on_tool_error_callback) and `repeat_call_guard` (before_tool_callback) | `guards.py` |

The step budget (`--max-calls`) stays in place as the last line of defense. In offline testing, a
scripted model that ignored the repeat guard's message still hit the budget, which is the point of
having both layers.

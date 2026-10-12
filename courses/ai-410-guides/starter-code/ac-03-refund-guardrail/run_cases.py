"""
Replay every recorded tool call through your issue_refund, in order.
Paste the output into REPORT.md.

    python run_cases.py
"""

import json
import pathlib

import refunds
import review_queue
import store_data

CALLS = json.loads((pathlib.Path(__file__).parent / "recorded_calls.json").read_text())


def main() -> None:
    store_data.reset()
    review_queue.reset()
    print(f"AUTO_REFUND_LIMIT = ${refunds.AUTO_REFUND_LIMIT:.2f}\n")
    counts = {"refunded": 0, "queued": 0, "error": 0}
    money_out = 0.0
    for n, call in enumerate(CALLS, 1):
        result = refunds.issue_refund(call["args"], call["customer_id"])
        kind = "error" if "error" in result else result.get("status", "?")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "refunded":
            money_out += result["amount"]
        print(f"{n:2}. {call['customer_id']} said: {call['said']}")
        print(f"    args:   {call['args']}")
        print(f"    result: {result}\n")
    print(f"Summary: {counts['refunded']} refunded (${money_out:.2f} out), "
          f"{counts['queued']} queued for review, {counts['error']} rejected")
    for job in review_queue.JOBS:
        print(f"  {job['job_id']}: {job['payload']}")


if __name__ == "__main__":
    main()

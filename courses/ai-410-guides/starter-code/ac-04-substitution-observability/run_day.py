"""
Run one day of Puget Pantry orders through your instrumented feature.
Paste the output into REPORT.md.

    python run_day.py
"""

import instrumented
import obs
import push
from pantry_data import ORDERS


def main() -> None:
    obs.reset()
    push.reset()
    pages = []
    for order in ORDERS:
        trace = instrumented.run_order(order)
        chooses = trace.spans_named("choose_substitute")
        line = f"{order['order_id']} {order['customer_id']}: "
        if not chooses:
            line += "nothing out of stock"
        for span, score in zip(chooses, trace.scores):
            line += f"[{span.attrs['item']} -> {span.attrs['chosen']}, {span.attrs['model_ms']} ms, quality {score['value']}] "
        paged = instrumented.should_page(trace)
        if paged:
            pages.append(order["order_id"])
        print(line + ("  ** PAGE **" if paged else ""))

    scores = [s["value"] for t in obs.TRACES for s in t.scores]
    slowest = max((s.attrs["model_ms"] for t in obs.TRACES for s in t.spans_named("choose_substitute")), default=0)
    print(f"\n{len(obs.TRACES)} orders, {len(scores)} substitutions, {len(push.SENT)} push notifications sent")
    print(f"Average substitution quality: {sum(scores) / len(scores):.2f}")
    print(f"Quality 0.0: {scores.count(0.0)} | 0.5: {scores.count(0.5)} | 1.0: {scores.count(1.0)}")
    print(f"Slowest model call: {slowest} ms")
    print(f"Pages: {len(pages)} {pages}")


if __name__ == "__main__":
    main()

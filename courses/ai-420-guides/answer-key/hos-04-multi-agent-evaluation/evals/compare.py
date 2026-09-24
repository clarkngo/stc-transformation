"""Side-by-side report of the single-agent and team runs, as a Markdown table.

    python evals/compare.py > evals/results/report.md
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
runs = {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json")) if p.stem in ("single", "team")}
if not runs:
    raise SystemExit("No results yet. Run evals/run_eval.py --system single (and --system team) first.")

rows = [("Success rate", "success_rate", "{:.0%}"), ("Cost per task (US$)", "cost_per_task_usd", "${:.5f}"),
        ("Tokens per task", "tokens_per_task", "{:,}"), ("Latency p50 (s)", "latency_p50_s", "{}"),
        ("Latency p95 (s)", "latency_p95_s", "{}"), ("Crashed scenarios", "crashes", "{}"),
        ("Scenarios run", "scenarios", "{}")]
names = list(runs)
print("| Metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
for label, key, fmt in rows:
    print(f"| {label} | " + " | ".join(fmt.format(runs[n]["summary"][key]) for n in names) + " |")

print("\n| Scenario | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
ids = [r["id"] for r in runs[names[0]]["results"]]
for sid in ids:
    cells = []
    for n in names:
        r = next((x for x in runs[n]["results"] if x["id"] == sid), None)
        cells.append("—" if r is None else ("PASS" if r["success"] else "FAIL"))
    print(f"| {sid} | " + " | ".join(cells) + " |")

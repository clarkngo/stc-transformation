"""Stage 4 (by hand): how often does the LLM judge agree with a human?

1. Read at least 6 transcripts in evals/results/<system>.json yourself.
2. Record your own pass/fail in evals/human_labels.json, e.g. {"order-status": true, "vhf-price": false}
3. python evals/agreement.py --system team

An automated judge is only worth trusting to the extent it agrees with careful human grading.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--system", choices=["single", "team"], required=True)
args = parser.parse_args()

labels = json.loads((HERE / "human_labels.json").read_text())
results = {r["id"]: r for r in json.loads((HERE / "results" / f"{args.system}.json").read_text())["results"]}

shared = [sid for sid in labels if sid in results]
if not shared:
    raise SystemExit("None of your labeled scenario ids appear in the results file.")

agree = [sid for sid in shared if labels[sid] == results[sid]["success"]]
print(f"Judge agreed with you on {len(agree)}/{len(shared)} scenarios ({len(agree) / len(shared):.0%}).")

# Which way does it err? A lenient judge inflates success rate; a strict one hides real wins.
lenient = [s for s in shared if results[s]["success"] and not labels[s]]
strict = [s for s in shared if not results[s]["success"] and labels[s]]
print(f"Judge passed, you failed (too lenient): {lenient or 'none'}")
print(f"Judge failed, you passed (too strict):  {strict or 'none'}")
for sid in lenient + strict:
    print(f"\n{sid}\n  judge said: {results[sid]['reason']}")

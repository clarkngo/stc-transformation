"""Offline smoke test for AI 420 HOS 1. Run from inside a copy of the HOS folder.

    python smoke_hos1.py answer|starter
Real network calls to Wikipedia; the Gemini model is replaced by a scripted stand-in.
"""
import json, os, sys, subprocess
from google.genai import types

variant = sys.argv[1]
sys.path.insert(0, os.getcwd())
os.environ.setdefault("GOOGLE_API_KEY", "offline-smoke-test")
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

from research import tools, baseline
r = tools.wiki_search("Space Needle"); check("wiki_search live", r["status"] == "ok" and r["titles"], r.get("titles"))
r = tools.wiki_summary("Space Needle"); check("wiki_summary live", r["status"] == "ok" and "1962" in r["summary"], r.get("title"))
r = tools.wiki_summary("No Such Article Zzqx"); check("wiki_summary missing article -> error dict", r["status"] == "error")
check("calculator", tools.calculator("2007 - 1962") == {"status": "ok", "result": 45})
check("calculator rejects code", tools.calculator("__import__('os')")["status"] == "error")

# Every task's expected answer should be reachable from the intros the tools return.
tasks = json.load(open("tasks.json"))
check("tasks.json has 5 tasks", len(tasks) == 5)


def resp(parts, tokens=10):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=tokens))


class FakeModels:
    def __init__(self, script): self.script, self.calls = list(script), 0
    def generate_content(self, model, contents, config=None):
        self.calls += 1
        item = self.script.pop(0) if self.script else ("text", "done")
        if item[0] == "call":
            return resp([types.Part(function_call=types.FunctionCall(name=item[1], args=item[2]))])
        return resp([types.Part(text=item[1])])


class FakeClient:
    def __init__(self, script): self.models = FakeModels(script)


fake = FakeClient([("text", "The Space Needle was completed in 1962.")])
baseline.get_client = lambda: fake
check("baseline runs one call", baseline.run("q")["steps"] == 1 and fake.models.calls == 1)

if variant == "starter":
    check("react_loop.py is left for the student", not os.path.exists("research/react_loop.py"))
    out = subprocess.run([sys.executable, "-c", "import compare"], capture_output=True, text=True,
                         env={**os.environ, "GOOGLE_API_KEY": ""})
    check("compare.py fails clearly before Stage 1", out.returncode != 0 and "react_loop" in (out.stderr + out.stdout),
          (out.stderr.strip().splitlines() or [""])[-1][:90])
else:
    from research import react_loop, plan_execute
    fake = FakeClient([("call", "wiki_summary", {"title": "Space Needle"}),
                       ("call", "calculator", {"expression": "2007 - 1962"}),
                       ("text", "45 years.")])
    react_loop.get_client = lambda: fake
    r = react_loop.run("How many years…?")
    check("react loop: tools executed, observation fed back, final answer", r["answer"] == "45 years." and r["tool_calls"] == 2 and r["steps"] == 3,
          f"steps={r['steps']} tool_calls={r['tool_calls']}")
    check("react loop: real tool result recorded", any(e.get("result", {}).get("result") == 45 for e in r["log"]))
    fake = FakeClient([("call", "no_such_tool", {})] + [("call", "calculator", {"expression": "1+1"})] * 10)
    react_loop.get_client = lambda: fake
    r = react_loop.run("loop forever", max_steps=2)
    check("react loop: unknown tool returns an error observation", any(e.get("tool") == "no_such_tool" and e["result"]["status"] == "error" for e in r["log"]))
    check("react loop: step budget stops honestly", r["steps"] == 2 and "couldn't finish" in r["answer"])
    plan_fake = FakeClient([("text", json.dumps(["Find when the Space Needle was completed", "Subtract from 2007"])),
                            ("text", "45 years.")])
    step_fake = FakeClient([("text", "1962"), ("text", "45")])
    plan_execute.get_client = lambda: plan_fake
    react_loop.get_client = lambda: step_fake
    r = plan_execute.run("How many years…?")
    check("plan-and-execute: plan made, each step run, synthesized", r["plan"] and len(r["plan"]) == 2 and r["answer"] == "45 years.",
          f"steps={r['steps']}")

# The ADK version loads (adk web imports it the same way).
import adk_research_agent.agent as a
check("ADK agent loads with 3 tools", len(a.root_agent.tools) == 3, a.root_agent.name)

# Missing key: friendly message, not a traceback.
out = subprocess.run([sys.executable, "-m", "research.baseline", "hi"], capture_output=True, text=True,
                     env={k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"} | {"GOOGLE_API_KEY": ""})
check("missing key -> friendly message", "GOOGLE_API_KEY is not set" in (out.stdout + out.stderr), (out.stdout + out.stderr).strip()[:80])

print("\nHOS 1", variant, "->", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)

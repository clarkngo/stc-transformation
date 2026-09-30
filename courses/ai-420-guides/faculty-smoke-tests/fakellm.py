"""A scripted stand-in for Gemini that ADK agents can run on offline.

policy(llm_request, call_index) -> ("call", name, args) | ("text", str)
"""
import os, sys
from typing import AsyncGenerator
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

sys.path.insert(0, os.getcwd())
os.environ.setdefault("GOOGLE_API_KEY", "offline-smoke-test")

OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def last_function_responses(req):
    """Function responses in the most recent user turn of the request."""
    for c in reversed(req.contents or []):
        parts = [p for p in (c.parts or []) if p.function_response]
        if parts:
            return {p.function_response.name: p.function_response.response for p in parts}
        if c.role == "user" and any(p.text for p in (c.parts or [])):
            return {}
    return {}


class ScriptedLlm(BaseLlm):
    model: str = "scripted-stand-in"
    policy: object = None
    calls: int = 0
    requests: list = []

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        self.requests.append(llm_request)
        item = self.policy(llm_request, self.calls)
        if item[0] == "call":
            part = types.Part(function_call=types.FunctionCall(name=item[1], args=item[2]))
        else:
            part = types.Part(text=item[1])
        yield LlmResponse(content=types.Content(role="model", parts=[part]),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(
                              prompt_token_count=100, candidates_token_count=20, total_token_count=120))


def script(steps):
    """A policy that plays back a fixed list of steps, then says 'Done.'."""
    steps = list(steps)
    return lambda req, n: steps[n - 1] if n <= len(steps) else ("text", "Done.")


async def run_agent(agent, message, session_id="s", max_calls=12, state=None):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.agents.run_config import RunConfig
    svc = InMemorySessionService()
    await svc.create_session(app_name="smoke", user_id="u", session_id=session_id, state=state or {})
    runner = Runner(app_name="smoke", agent=agent, session_service=svc)
    events = []
    async for ev in runner.run_async(user_id="u", session_id=session_id,
                                     new_message=types.Content(role="user", parts=[types.Part(text=message)]),
                                     run_config=RunConfig(max_llm_calls=max_calls)):
        events.append(ev)
    session = await svc.get_session(app_name="smoke", user_id="u", session_id=session_id)
    return events, session


def tool_results(events):
    out = []
    for ev in events:
        for p in (ev.content.parts if ev.content and ev.content.parts else []):
            if p.function_response:
                out.append((p.function_response.name, p.function_response.response))
    return out


def final_text(events):
    for ev in reversed(events):
        for p in (ev.content.parts if ev.content and ev.content.parts else []):
            if p.text:
                return p.text
    return None

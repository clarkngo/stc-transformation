"""Optional: send every agent run to Langfuse as a trace (same free account as AI 410 HOS 4).

If LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in .env, importing this
module instruments ADK so each model call and tool call becomes a span in
Langfuse. If they aren't set, nothing happens and the local trace printout
from run_scenarios.py still works.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

_client = None

if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor

    _client = get_client()
    GoogleADKInstrumentor().instrument()
    print("Langfuse tracing on:", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
else:
    print("Langfuse keys not set: local traces only (see traces/).")


def flush() -> None:
    """Send any buffered spans before the script exits."""
    if _client is not None:
        _client.flush()

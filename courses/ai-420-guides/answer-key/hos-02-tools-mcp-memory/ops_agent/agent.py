"""Harbor Supply Co. operations agent.

Tools come from two places:
  - the MCP server (mcp_server/server.py), started as a subprocess over stdio
  - plain Python function tools for memory (ops_agent/memory.py)
Short-term memory is ADK session state (set_current_customer); long-term memory is Chroma.

Run from the HOS folder:  adk web   (then pick "ops_agent")
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

from .memory import recall_customer_facts, remember_customer_fact

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def set_current_customer(customer_name: str, tool_context: Context) -> dict:
    """Remember who you're talking to for the rest of this conversation.

    Args:
        customer_name: The customer's full name, once they've told you.
    """
    tool_context.state["current_customer"] = customer_name  # short-term: this session only
    return {"status": "ok", "current_customer": customer_name}


harbor_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server" / "server.py")],
        ),
        timeout=30,
    ),
)

root_agent = Agent(
    name="ops_agent",
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    description="Operations assistant for Harbor Supply Co., a marine supply store.",
    instruction=(
        "You help the staff and customers of Harbor Supply Co., a marine supply store in the Seattle area.\n"
        "- When a customer gives their name, call set_current_customer, then recall_customer_facts.\n"
        "- For a single order, use order_status. For restocking, use low_stock_report. Use query_database "
        "only for questions those two can't answer.\n"
        "- For conditions on the water, use get_marine_weather. To read a web page, use fetch_page.\n"
        "- If you learn a lasting preference, save it with remember_customer_fact.\n"
        "- If a tool returns status 'error', read the error, fix your input, and try once more before giving up.\n"
        "Current customer (if known): {current_customer?}"
    ),
    tools=[harbor_tools, set_current_customer, remember_customer_fact, recall_customer_facts],
)

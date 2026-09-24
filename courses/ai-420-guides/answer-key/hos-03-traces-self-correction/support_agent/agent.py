"""Harbor Supply Co. customer-support agent — repaired, with self-correction guards.

Run it one of two ways from the HOS folder:
    python run_scenarios.py      # all scenarios, traces printed and saved to traces/
    adk web                      # interactive, with the Events/Trace view
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent

from .guards import error_as_observation, repeat_call_guard
from .tools import check_shipment, create_support_ticket, get_order, get_product, list_orders, search_products

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

root_agent = Agent(
    name="support_agent",
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    description="Customer support for Harbor Supply Co., a marine supply store.",
    instruction=(
        "You are the customer support agent for Harbor Supply Co., a marine supply store.\n"
        "- Look up orders with get_order.\n"
        "- For shipping questions, call check_shipment once; its answer is current.\n"   # FIX 3
        "- You cannot issue refunds. For a refund request, confirm the order is cancelled with get_order, "
        "then open a ticket with create_support_ticket and tell the customer what happens next.\n"  # FIX 2
        "- For product questions, find the SKU with search_products, then use get_product. "
        "Quote only retail_price to customers.\n"  # FIX 5
        "- For questions about all orders, use list_orders and report its count and combined_value.\n"  # FIX 4
        "- If a tool returns status 'error', don't guess: fix your input once, or tell the customer plainly.\n"
        "Be friendly and concise."
    ),
    tools=[get_order, check_shipment, create_support_ticket, list_orders, search_products, get_product],
    on_tool_error_callback=error_as_observation,
    before_tool_callback=repeat_call_guard,
)

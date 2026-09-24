"""Harbor Supply Co. customer-support agent.

Run it one of two ways from the HOS folder:
    python run_scenarios.py      # all scenarios, traces printed and saved to traces/
    adk web                      # interactive, with the Events/Trace view
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent

from .tools import check_shipment, get_order, get_price, get_product, list_orders, search_products

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

root_agent = Agent(
    name="support_agent",
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    description="Customer support for Harbor Supply Co., a marine supply store.",
    instruction=(
        "You are the customer support agent for Harbor Supply Co., a marine supply store.\n"
        "- Look up orders with get_order.\n"
        "- For shipping questions, always confirm the latest tracking status with check_shipment before answering.\n"
        "- If a customer asks for a refund on a cancelled order, process it with issue_refund.\n"
        "- For product questions, find the SKU with search_products, then look it up.\n"
        "- For questions about all orders, use list_orders.\n"
        "Be friendly and concise."
    ),
    tools=[get_order, check_shipment, list_orders, search_products, get_price, get_product],
)

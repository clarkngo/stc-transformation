"""Harbor Supply Co. support, as a team: a coordinator plus three specialists.

The coordinator never calls tools itself. It reads the customer's message and
transfers to the specialist that owns that kind of question (ADK's LLM-driven
delegation via sub_agents, the same pattern as agent-development hos07/hos08).
Each specialist has only the tools it needs.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from single_agent.guards import error_as_observation, repeat_call_guard  # noqa: E402
from single_agent.tools import (  # noqa: E402
    check_shipment, create_support_ticket, get_order, get_product, list_orders, search_products)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GUARDS = {"on_tool_error_callback": error_as_observation, "before_tool_callback": repeat_call_guard}

orders_agent = Agent(
    name="orders_agent",
    model=MODEL,
    description="Handles a customer's own orders: status, shipping and tracking, and refund requests.",
    instruction=(
        "You handle order questions for Harbor Supply Co. Use get_order, then check_shipment once for shipping. "
        "You cannot issue refunds: for a cancelled order, open a ticket with create_support_ticket and say what happens next. "
        "If an order isn't found, ask the customer to double-check the number. "
        "If the customer also asks about products or store-wide numbers, transfer back to coordinator."
    ),
    tools=[get_order, check_shipment, create_support_ticket],
    **GUARDS,
)

catalog_agent = Agent(
    name="catalog_agent",
    model=MODEL,
    description="Answers product questions: what the store sells, retail prices, and stock.",
    instruction=(
        "You answer product questions for Harbor Supply Co. Find SKUs with search_products, then use get_product. "
        "Quote only retail_price. Never share wholesale or internal costs. "
        "If the customer also asks about an order, transfer back to coordinator."
    ),
    tools=[search_products, get_product],
    **GUARDS,
)

reports_agent = Agent(
    name="reports_agent",
    model=MODEL,
    description="Store-wide numbers for staff: order counts and combined sales value.",
    instruction="You answer store-wide questions with list_orders. Report its count and combined_value exactly.",
    tools=[list_orders],
    **GUARDS,
)

root_agent = Agent(
    name="coordinator",
    model=MODEL,
    description="Front desk for Harbor Supply Co. support.",
    instruction=(
        "You are the front desk for Harbor Supply Co. support. Don't answer questions yourself: transfer to "
        "orders_agent for a customer's own orders, catalog_agent for products and prices, or reports_agent "
        "for store-wide numbers. If a message has two kinds of questions, handle one specialist at a time "
        "and make sure both get answered."
    ),
    sub_agents=[orders_agent, catalog_agent, reports_agent],
)

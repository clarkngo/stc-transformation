"""
The Cascade Family Clinic chat agent — already written, you don't need
to change it. It's the HOS 1 agentic loop, plus two things:

  1. The logged-in patient is fixed when you start the app
     (--patient P1001). Every tool call gets THAT patient_id, whatever
     the model asks for. The model never chooses whose appointment
     it's acting on.
  2. Every tool call is printed, so your transcript shows which tool
     ran with which arguments.

Run:  python agent.py --patient P1001
"""

import argparse
import inspect

from dotenv import load_dotenv
from google import genai

from tools import TOOLS, TOOL_FUNCTIONS
import clinic_data

load_dotenv()
MODEL = "gemini-3.8-flash"
MAX_TURNS = 5  # hard ceiling so a misbehaving loop can't run forever


def system_instruction(patient_id: str) -> str:
    patient = clinic_data.PATIENTS[patient_id]
    return (
        "You are the scheduling assistant for Cascade Family Clinic. "
        f"You are talking with {patient['name']}. "
        f"Today is {clinic_data.now():%A, %B %d, %Y}, and the time is "
        f"{clinic_data.now():%I:%M %p}. Use your tools to look up and change "
        "appointments; never invent slots or confirmations. You do not give "
        "medical advice."
    )


def run_tool(name: str, arguments: dict, patient_id: str) -> dict:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"No tool named '{name}'."}
    args = dict(arguments)
    if "patient_id" in inspect.signature(fn).parameters:
        args["patient_id"] = patient_id  # the app decides, not the model
    print(f"  -> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
    try:
        result = fn(**args)
    except NotImplementedError:
        result = {"error": f"{name} isn't implemented yet."}
    except TypeError as e:  # model sent arguments the function doesn't take
        result = {"error": f"Bad arguments for {name}: {e}"}
    print(f"  <- {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="P1001", choices=sorted(clinic_data.PATIENTS))
    patient_id = parser.parse_args().patient

    client = genai.Client()
    instruction = system_instruction(patient_id)
    previous_id = None  # carries the conversation from one message to the next
    print(f"Logged in as {clinic_data.PATIENTS[patient_id]['name']} ({patient_id}). Type 'quit' to exit.")

    while True:
        message = input("\nyou> ").strip()
        if message.lower() in {"quit", "exit"}:
            break
        interaction = client.interactions.create(
            model=MODEL, input=message, tools=TOOLS,
            system_instruction=instruction, previous_interaction_id=previous_id,
        )
        for _ in range(MAX_TURNS):
            calls = [step for step in interaction.steps if step.type == "function_call"]
            if not calls:
                break
            results = [
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": str(run_tool(call.name, call.arguments, patient_id))}],
                }
                for call in calls
            ]
            interaction = client.interactions.create(
                model=MODEL, input=results, tools=TOOLS,
                system_instruction=instruction, previous_interaction_id=interaction.id,
            )
        previous_id = interaction.id
        print(f"agent> {interaction.output_text}")


if __name__ == "__main__":
    main()

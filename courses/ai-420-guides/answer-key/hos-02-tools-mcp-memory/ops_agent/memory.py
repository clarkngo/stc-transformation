"""Long-term memory: facts the agent chooses to save, recalled later by meaning.

Stored in a local Chroma database under .memory/ (no account, no sign-up).
Survives restarts, unlike session state, which lasts one conversation.
The first call downloads Chroma's small embedding model (~80 MB) once.
"""

import time
from pathlib import Path

import chromadb

PATH = Path(__file__).resolve().parent.parent / ".memory"
_collection = None


def _store():
    global _collection
    if _collection is None:
        _collection = chromadb.PersistentClient(path=str(PATH)).get_or_create_collection("harbor_memory")
    return _collection


def remember_customer_fact(customer_name: str, fact: str) -> dict:
    """Save a lasting fact about a customer for future conversations (a preference, their boat, a standing request).

    Only save things that will still matter next time. Never save payment details or passwords.

    Args:
        customer_name: The customer's full name, e.g. "Ana Reyes".
        fact: One short fact, e.g. "Prefers email updates over phone calls".
    """
    _store().add(
        ids=[f"{customer_name}-{time.time_ns()}"],
        documents=[fact],
        metadatas=[{"customer": customer_name.strip().lower()}],
    )
    return {"status": "ok", "saved": fact}


def recall_customer_facts(customer_name: str, topic: str = "") -> dict:
    """Recall saved facts about a customer. Call this at the start of any conversation with a known customer.

    Args:
        customer_name: The customer's full name.
        topic: Optional topic to focus on, e.g. "shipping" or "their boat".
    """
    store = _store()
    where = {"customer": customer_name.strip().lower()}
    if store.count() == 0:
        return {"status": "ok", "facts": []}
    result = store.query(query_texts=[topic or customer_name], n_results=5, where=where)
    return {"status": "ok", "facts": result["documents"][0] if result["documents"] else []}

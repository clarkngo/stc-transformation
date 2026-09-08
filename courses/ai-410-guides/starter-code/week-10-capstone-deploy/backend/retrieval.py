"""
Retrieval for the RAG pipeline. Instrumented (Week 9, solved).
"""

import os

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langfuse.decorators import observe

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024


@observe()
def retrieve(query: str, k: int = 5) -> list[str]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    q_vec = result.embeddings[0].values

    chroma = chromadb.HttpClient(
        host=os.environ.get("CHROMA_HOST", "localhost"),
        port=int(os.environ.get("CHROMA_PORT", 8001)),
    )
    collection = chroma.get_or_create_collection(name="documents")
    results = collection.query(query_embeddings=[q_vec], n_results=k)
    return results["documents"][0]

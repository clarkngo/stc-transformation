"""
AI 410 — Week 7 starter: Asynchronous Data Processing

Chat (with RAG + guardrails) is unchanged from Weeks 4-6. This week
adds document upload — currently synchronous (blocks the request).
Your job: make it async using the queue, and add a status endpoint.

Remember: run the worker in a SEPARATE terminal —
    python worker.py
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import run_agent
from jobs import ingest_document_job
from queue_setup import queue

load_dotenv()

app = FastAPI(title="AI 410 — Week 7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class UploadRequest(BaseModel):
    doc_path: str  # path to a file already on the server for this exercise


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    return ChatResponse(reply=run_agent(body.message))


@app.post("/documents")
def upload_document(body: UploadRequest):
    # TODO(week7): this currently runs ingestion INLINE, blocking the
    # request until it finishes. Replace the line below with:
    #
    #   job = queue.enqueue(ingest_document_job, body.doc_path)
    #   return {"job_id": job.id}
    #
    # so the endpoint returns immediately instead of waiting.
    result = ingest_document_job(body.doc_path)
    return {"result": result}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    # TODO(week7): implement this using rq's job fetching, e.g.:
    #
    #   from rq.job import Job
    #   from queue_setup import redis_conn
    #   job = Job.fetch(job_id, connection=redis_conn)
    #   return {"status": job.get_status(), "result": job.result}
    raise NotImplementedError("implement job_status() — see the TODO above")


@app.get("/health")
def health():
    return {"status": "ok"}

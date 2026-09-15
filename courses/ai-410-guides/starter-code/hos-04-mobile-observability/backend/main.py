"""
HOS 3 answer key: Guardrails & Asynchronous Processing.

Chat (with RAG + guardrails) is unchanged in shape from HOS 1-2. This
HOS adds document upload as a background job instead of blocking the
request, plus a status endpoint to poll it.

Remember: run the worker in a SEPARATE terminal — python worker.py
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from rq.job import Job

from agent import run_agent
from jobs import ingest_document_job
from queue_setup import queue, redis_conn

load_dotenv()

app = FastAPI(title="AI 410 — HOS 3")


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
    job = queue.enqueue(ingest_document_job, body.doc_path)
    return {"job_id": job.id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)
    return {"status": job.get_status(), "result": job.result}


@app.get("/health")
def health():
    return {"status": "ok"}

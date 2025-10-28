# Proposal: Deploying OpenAI’s gpt-oss via Ollama for (School Name)

Short deliverable: a concise blog-style proposal, a quick-reference hardware table, example architectures (chat, code-assist, RAG), estimated throughput assumptions, a cost-comparison vs hosted API, recommended alternatives, an ops checklist, and next steps.

---

## Executive summary

Goal: run `gpt-oss` locally with Ollama to provide student-facing LLM services (chat assist, code assist) for up to ~50 concurrent users with data privacy and low-latency inference.

Recommendation in brief:
- Start with a mid-tier deployment supporting ~50 concurrent users using a quantized 13B–33B model on 1–2 GPUs (48–80 GB aggregate GPU RAM) for chat and small code tasks. Use a dedicated higher-memory GPU (80 GB-class, or 2×48GB NVLink) for heavier code models (70B or code-specialized models).
- Use RAG with an internal vector DB (PGVector, Milvus, or Weaviate) for course-specific knowledge — keep embeddings and chunks on-prem for privacy.
- Evaluate cloud burst capacity for peak load and for heavy model fine-tuning or experimentation.

Why Ollama + gpt-oss: Ollama provides a local inference runtime, model packaging, and an easy developer UX (model registry, local serving). `gpt-oss` is an open-source GPT-like family that reduces recurring API costs and keeps student data private on-prem.

Scope & exclusions:
- Scope: chat assist, code assist, RAG with private corpora, ops and monitoring guidance.
- Exclude: DeepSeek (explicitly requested), third-party hosted vector indexing services unless used for cloud-bursting only.

---

## Contract (inputs/outputs, success criteria)
- Inputs: user prompts (text, code), optional uploaded context/docs, vector DB queries for RAG.
- Outputs: text responses, code completions, citations to source documents (when RAG used).
- Success criteria: median token-latency ≤ 700ms for chat (short prompts), p95 latency ≤ 2s for interactive chats for up to 50 concurrent users; privacy: no outgoing user PII to third-party APIs; budget: lower total cost of ownership (TCO) vs hosted API at sustained usage > X tokens/month (see cost section).

Edge cases to plan for:
- Very long context windows (must chunk and embed); bursty concurrency > 50 (use cloud burst or queueing). 
- Model drift or hallucination in grading contexts (use RAG + citations + guardrails).
- Sensitive content or student data (access control, audit logs, retention policy required).

---

## Quick-reference hardware table (short)

Notes before table:
- Model sizes and memory behaviors vary with quantization and runtime. Values below are estimates and assume optimized runtimes (4-bit quantization when possible, Ollama support for quantized formats, or GGML builds for CPU fallback).
- ``GPU RAM`` = dedicated GPU memory available to the model. Add ~20% headroom for system memory and runtime.

| Tier | Target model size (est) | On-prem GPU suggestion | Minimum GPU RAM | vCPU / RAM | Storage (nvme) | Network | Typical use-case |
|---|---:|---|---:|---:|---:|---:|---|
| Small (POC / dev) | 7B (quantized) | 1× NVIDIA L4 / A10G | 24–32 GB | 8–16 vCPU, 64–128 GB RAM | 1 TB NVMe | 1 Gbps | Developer testing, instructor demos, small classroom (≤10 concurrent)
| Medium (production interactive) | 13B–33B (4-bit) | 1× NVIDIA L4 / A10G or 1× 48–80 GB GPU | 48 GB | 16–32 vCPU, 128–256 GB RAM | 2–4 TB NVMe | 10 Gbps | 50 concurrent chat/code assist (latency-sensitive)
| Heavy (higher-quality/code) | 70B+ (quantized) | 1× A100 80GB / H100 80GB or 2× 48–80GB NVLink | 80+ GB (or NVLink pooled) | 32+ vCPU, 256+ GB RAM | 4+ TB NVMe | 25+ Gbps | Code assist, larger context windows, small multi-user clusters
| Scale-out (high throughput) | Multi-node | Cluster of 2–4 A100/H100 or multi-CPU nodes + shards | 80+ GB per GPU/node | 64+ vCPU per node, 512+ GB total RAM | 10+ TB NVMe across cluster | 25–100 Gbps, RDMA | Serving 100s of concurrent users with batching & autoscaling

Cloud equivalence guidance (vendor families):
- AWS: g5 / g5n (A10G), p4d/p4de (A100), p5 (H100)
- Azure: NC/Dv4/ND A100/H100 series (ND A100 v4)
- GCP: A2 (A100) and next-gen H100 instances

Storage and DB: Use NVMe for model storage and local cache. For vector DB use dedicated nodes (Milvus/MongoDB+PGVector/Weaviate) on fast NVMe and 10Gb+ network.

---

## Throughput & latency assumptions (estimates and how to reason about them)

Assumptions:
- Interactive chat prompt size: 32–256 tokens. Response size: 32–256 tokens. 
- Token-generation throughput depends on model size, GPU, quantization, and batching.
- Concurrent users measured as concurrent active request streams (not session count).

Rough estimates (example, conservative):
- 7B quantized on L4/A10G: ~80–250 tokens/s single-stream; latency per 64-token response: ~100–400ms.
- 13B quantized on 48GB GPU: ~30–120 tokens/s; 64-token response latency: ~400–900ms.
- 70B on A100 80GB (optimized, NVLink): ~10–40 tokens/s; 64-token response latency: ~1.5–6s.

Concurrency mapping (ballpark):
- 50 concurrent users with short responses (avg 64 tokens):
  - 13B on 48GB GPU: batch scheduling + streaming can handle ~40–80 concurrent if requests are staggered; aim for 1 GPU + small batcher + request queue. 
  - 70B: 50 concurrent likely needs multiple GPUs or queuing — expect p95 latencies > 2s unless scaled out.

How to measure and tune: start with internal microbenchmarks (measure tokens/s for your chosen model and prompt profile) then compute: required tokens/s = concurrent_users × avg_response_tokens × (1 / target_latency_seconds).

Example compute: target p95 latency 2s, avg response 128 tokens; tokens/s needed for 50 users = 50 * 128 / 2 = 3,200 tokens/s. Then pick a GPU configuration whose tokens/s meets that at acceptable latency.

---

## Cost comparison: on-prem / cloud GPU vs hosted API (example calculation)

Important: pricing changes quickly; below are illustrative example calculations to compare recurring hosted API vs owning GPU capacity. Replace numbers with vendor quotes for procurement accuracy.

Model of comparison: sustained interactive usage: 50 concurrent users, average 128 tokens generated per response, 30 responses per user-hour (approx 1 response every 2 minutes) → tokens per hour = 50 * 30 * 128 = 192,000 tokens/hr → ~4.6M tokens/day (~138M tokens/month)

Hosted-API example (hosted LLM provider priced per 1K tokens):
- If a hosted API costs $0.03 per 1K tokens (example), monthly cost = 138,000 × $0.03 = $4,140 (note: this is illustrative — check provider pricing). If using higher-cost models or larger response volumes, costs scale linearly.

Cloud GPU (on-demand) example (illustrative ranges):
- 1× A100 80GB on-demand: approximate $10–35/hr (varies by cloud & region). Using $20/hr → monthly cost (24/7) = $14,400. But you can schedule/scale (not 24/7), share across groups, and amortize procurement vs on-demand.
- Smaller instance (g5 A10G) might be $1–4/hr; if that suffices for your load, monthly costs drop significantly.

On-prem purchase example:
- One A100 80GB GPU card (or equivalent) plus server, networking, power, maintenance — rough CAPEX ballpark: $40K–120K depending on node (including CPU, RAM, storage). Amortize over 3 years → $1.1K–3.3K/month plus ops & power.

Interpretation:
- For low, bursty usage, hosted API often costs less upfront and has simpler ops.
- For sustained medium-to-high usage (hundreds of millions tokens/month), owning GPUs or reserved cloud instances becomes cost-effective.
- Also factor in indirect costs: staff time, security/compliance, and the value of keeping student data on-prem.

Recommendation: run a 30–60 day pilot, track tokens/month and model performance. Use that to decide whether cloud reserved instances or on-prem purchase pays off.

---

## Alternative models & when to use them

- LLaMA 2 (7B/13B/70B) — good general-purpose family, widely supported.
- Vicuna — instruction-tuned LLaMA forks, lower-cost and good for chat prototypes.
- Code Llama / StarCoder / CodeGen — specialized for code assist and completions (prefer for code tasks vs generic chat models).
- MPT-7B / MosaicML models — flexible license options and good throughput on CPUs for smaller tasks.
- Mistral / Mistral-Instruct — strong open models with good performance for chat.

When to pick:
- Chat (conversational help): 13B–33B with instruction-tuning or Vicuna derivatives.
- Code assist: Code Llama or StarCoder family; smaller code models often have better correctness for completions.
- RAG: any base model + an embedding model; for embeddings consider on-device cheaper embedder (e.g., open embedding models) to keep privacy.

---

## Example architectures (textual diagrams)

### 1) Simple chat service (single-node)
- Components: Web UI → API Gateway (auth) → Ollama inference runtime (local model) → Optional cache → Response streaming to client
- Data flow: User → Gateway (auth & rate-limit) → Ollama model → return stream
- Notes: Keep logs in protected store; apply content filters and rate limiting; store opt-in transcripts for training only with consent.

### 2) Code-assist service (recommended separation)
- Components: Web IDE / VS Code extension → Backend service (API) → Ollama runtime (code-specialized model) + code-indexer for workspace search → Optional ephemeral sandbox for running submitted code.
- Data flow: User code snippet → API + static analysis → model request → model returns suggested patch / explanation.
- Notes: Run automatic sandboxing for executed code; redact secrets before sending to model; provide "explain" and "suggest patch" modes.

### 3) RAG for course materials (recommended for accuracy)
- Components: UI → API → RAG orchestrator → Vector DB (Milvus/Weaviate/PGVector) + chunk store (S3/NVMe) → Embedding service (local) → Ollama model
- Data flow: User query → Orchestrator issues vector search → Top-K docs retrieved → Assembled prompt with citations → Ollama model generates answer with citations → return.
- Notes: Keep embeddings & vectors on-prem; add a provenance layer that stores source doc ids and snippet offsets for reproducible citations.

### Horizontal scaling pattern
- Front tier: API gateways behind LB (stateless) + autoscale.
- Worker tier: Ollama inference: scale by instances (1 per GPU) with queue + batching service.
- Storage tier: Vector DB cluster, archival S3, logging & metrics.

---

## Ops checklist (deployment & monitoring)

Pre-deployment
- Procurement: choose GPU tier based on pilot benchmarks (7B/13B/70B).
- Network: ensure 10Gb+ internal network for vector DB and NVMe access; secure per PCI/edu policies.
- Security: design RBAC, network segmentation, TLS everywhere, secrets management (HashiCorp Vault or cloud secrets manager).
- Privacy & compliance: approve data retention policy and student consent policy; ensure FERPA compliance where required.

Deployment
- Containerize or use Ollama host install; use systemd or k8s for lifecycle management.
- Use GPU device plugins for k8s (NVIDIA device plugin) if using k8s.
- Setup vector DB nodes on fast NVMe; use backups/snapshots nightly.

Monitoring & observability
- Metrics: request count, latency (p50/p95/p99), tokens/s, GPU utilization, memory usage, queue depth, embedding latency, vector DB latencies.
- Logs: structured request logs with pseudonymized user ids, model inputs hashed (if storing), error logs.
- Alerting: high GPU mem pressure, GPU/CPU saturation, p95 latency > threshold, vector DB errors.
- Tracing: distributed traces for end-to-end requests (API → model → DB).

Maintenance
- Model updates: plan scheduled windows for model upgrades and A/B tests.
- Retraining/fine-tuning: if needed, have a separate training environment; do not train in the same cluster as serving.
- Backups: config and vector DB backups daily; test restores quarterly.

Security hardening
- Egress controls: deny outgoing traffic to third-party LLM APIs by default.
- Input sanitization: redact secrets and PII automatically with a pre-filter before model requests.
- Rate limiting & quotas per user to prevent abuse.

Incident response
- Playbook for model hallucination causing harm: immediate rollback to previous model, block feature, notify owners.
- Audit trails enabled: who called what model and when (for compliance).

---

## Example pilot plan (30–60 days)
1. POC (Week 0–2): Choose a 13B model; run Ollama on a single 48GB GPU; implement a simple chat UI and instrument metrics.
2. Pilot (Week 3–6): Onboard 1–3 courses; collect usage metrics, token volumes, latency, and qualitative feedback.
3. Scale decision (Week 6–8): Decide whether to buy on-prem GPU(s), reserve cloud instances, or continue hosted API. If buying, procure hardware; if cloud, reserve instances or implement autoscaling.

KPIs to measure: average tokens/session, p95 latency, GPU utilization, monthly tokens, incidence of hallucinations requiring human review.

---

## Recommended next steps (concrete)
- Short-term (0–2 weeks): run microbenchmarks for candidate models (7B/13B/70B) using representative prompts; measure tokens/s and latency.
- Medium-term (2–8 weeks): run 30-day pilot with a 13B model on a 48GB GPU node, integrate with a minimal vector DB for course notes, and gather metrics.
- Decision point (after pilot): choose on-prem purchase vs reserved cloud capacity using measured token/month and latency needs.

Operational quick wins:
- Enforce egress-blocking network policy for inference nodes.
- Automate redaction of obvious PII before inference.
- Use a lightweight request queue + batcher to increase GPU throughput while maintaining streaming responses.

---

## Citations & further reading
- Ollama — official docs & guides: https://ollama.com
- Hugging Face: model hub and deployment docs — https://huggingface.co
- Meta LLaMA & Code Llama announcements: https://ai.meta.com/ (LLaMA 2 and Code Llama info)
- Stanford Alpaca (instruction-tuning example): https://crfm.stanford.edu/2023/03/13/alpaca.html
- Vector DB options & comparisons: Milvus (https://milvus.io), Weaviate (https://weaviate.io), PGVector (https://github.com/pgvector/pgvector)
- Ollama blog posts and community examples (search “Ollama university” for community posts and tutorials).

Note on university case studies: public university-grade production deployments are often reported as internal projects or high-level case studies; where available, consult vendor case studies on Hugging Face and Ollama (they publish enterprise/education examples). For reproducible academic examples of model fine-tuning and instruction-tuning see the `Alpaca` resources linked above.

---

## Appendix: Quick decision checklist for sizing
- Low-latency chat & 50 concurrent users: start with 13B on 48GB GPU + batching. If p95 latencies exceed 2s, move to 80GB-class or add a second GPU & horizontal scale.
- Code assist requiring high accuracy: choose code-specific models (Code Llama / StarCoder) and target 80GB-class GPU for larger-context or larger models.
- RAG required for accuracy: embed & store docs on-prem; use smaller embedding models locally and keep vector DB on fast NVMe.


---

## Files created
- `gpt-oss-ollama-proposal.md` — this document: blog-style proposal, hardware table, architectures, ops checklist, cost/throughput guidance, citations.

---

If you want, I can:
- Run microbenchmarks locally (if you have hardware) or provide a small Python/bench harness to measure tokens/s for a selected model via Ollama’s local API.
- Produce a short slide deck or a one-page handout adapted for procurement.
- Generate a sample k8s manifest or systemd unit for running Ollama inference nodes.

Which of those would you like next? Replace the placeholder (School Name) with your school if you want a version customized for internal distribution.
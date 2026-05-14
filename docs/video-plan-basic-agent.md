# Video production pack: ADK Masterclass · Example #1 — Building Your First Agent

**Companion to:** [Build AI Agents & Automate Workflows (Beginner to Pro)](https://youtu.be/P4VFL9nIaIA)  
**Source repo:** [imvickykumar/agent-development-kit-crash-course](https://github.com/imvickykumar/agent-development-kit-crash-course) — **Example 1: Basic Agent** (greeting agent: name in, personalized reply out)  
**Official ADK docs:** [ADK quickstart](https://google.github.io/adk-docs/get-started/quickstart)

**Anchor lesson:** Example #1 introduces Google’s three building blocks — **Agent**, **Session**, and **Runner** — while walking a first agent that greets the user by name.

**Working title (technical cut):** *Example #1 (Basic Agent) — 10 concepts the crash course assumes you know*

**Target length:** ~24–28 minutes (tight) · ~32–38 minutes (extended: +5 min Q&A, +3 min “where to go next”)

**Audience:** Beginners who cloned the repo or watched the long tutorial and hit “it worked once” / “import error” / “why do I need .env in every folder?”

**STC tie-in (optional closer):** Bridge content = teach the **seams** (keys, cwd, agent vs script), not only the happy path — aligns with *Google ADK demystified* on the STC Content Engine strategy page.

---

## Strategic picture (STC · ADK Masterclass)

- **Democratization thought leadership:** STC-branded follow-on content can go *wider* than line-by-line competition with Google’s masterclass for senior developers — speak to managers, partners, and policy owners who will govern agents beside engineers, using workplace stories and failure modes.
- **Why Example #1:** The greeting agent carries the philosophical questions non-developers ask (intent, instructions, accountability) without requiring code on screen.
- **Execution risk:** A consistent on-camera anchor (Dean or named faculty) over years matters more than production polish; compare CS50’s single visible instructor model.

**Deliverables on the HTML pack:** [Strategy + action shortcuts](../adk-basic-agent-video-plan.html#strategic-framing) · [Full script on concepts page](../adk-basic-agent-concepts-interplay.html#script)

---

## 1. Learning outcomes

By the end, the viewer can:

1. Name Google’s three lesson blocks — **Agent**, **Session**, **Runner** — and say what each owns in plain language.
2. Explain what an **ADK agent** is versus a one-off “call Gemini from Python.”
3. Navigate the **Basic Agent** folder and name the files that define model + behavior + entrypoint.
4. Create/activate **venv**, install deps, and configure **`.env`** from `.env.example` **in the correct folder**.
5. Run the example and **diagnose** at least three common failures (key, Python version, wrong directory).
6. Describe **sessions** at a high level (why multi-turn works without manual history concatenation).

---

## 2. Pre-production checklist

| Task | Owner | Done |
|------|--------|------|
| Lock Example 1 folder name/path in current repo revision | | ☐ |
| Record `requirements.txt` / ADK package versions used | | ☐ |
| Test clean machine path: new venv → install → run | | ☐ |
| Prepare **deliberate failure** clips (wrong cwd, missing .env, bad key) | | ☐ |
| Screenshot file tree + `.env.example` for B-roll | | ☐ |
| Description draft with links (video, repo, ADK docs, disclaimers) | | ☐ |
| Chapters JSON / timestamps after edit | | ☐ |
| Captions export + technical term pass (ADK, Gemini, `.env`) | | ☐ |

**Disclaimers (say + put in description):** Not official Google training; APIs, model IDs, and pricing change; use keys only on machines you control.

---

## 3. Chapter map (YouTube chapters)

| # | Chapter title | ~Start |
|---|----------------|--------|
| 0 | Cold open — “if it broke, it’s one of these” | 0:00 |
| 1 | What this video is (and isn’t) | 0:45 |
| 2 | Repo map — why Example 1 is its own project | 2:00 |
| 3 | Concepts 1–2 — Agent vs script · Runtime mental model | 4:00 |
| 4 | Concepts 3–5 — venv, pip, Python version | 8:00 |
| 5 | Concepts 6–7 — `.env` per folder · Keys & billing (plain language) | 12:00 |
| 6 | Concepts 8–9 — cwd · Model IDs | 16:00 |
| 7 | Concept 10 — Instructions vs user message · intro to sessions | 19:30 |
| 8 | Live run + fix one bug on camera | 22:00 |
| 9 | Close — next step (Tool Agent) + STC bridge line | 25:00 |

*Adjust timestamps after record; extended cut adds “FAQ rapid fire” after ch. 8.*

---

## 4. Shot list (technical)

| Shot ID | Type | Subject | Notes |
|---------|------|---------|--------|
| S01 | Screen | Terminal `python --version` | Large font, high contrast |
| S02 | Screen | `ls` / file tree of repo root | Highlight `examples/` or equivalent path |
| S03 | Screen | Open Basic Agent folder in editor | Blur any real API key if visible |
| S04 | Screen | `.env.example` → rename to `.env` | Use dummy key for demo |
| S05 | Screen | `source .venv/bin/activate` (or Win equivalent) | Show prompt change |
| S06 | Screen | `pip install -r requirements.txt` | Optional: show `-q` vs full log tradeoff |
| S07 | Screen | Successful agent reply | Happy path |
| S08 | Screen | **Failure** — run from wrong directory | Stack trace readable |
| S09 | Screen | **Failure** — missing `.env` | Error readable |
| S10 | Screen | **Failure** — invalid key | Generic error, don’t leak key |
| S11 | B-roll | Static diagram: Agent vs raw API | Simple boxes + arrows |
| S12 | B-roll | Static: “per-example `.env`” | One slide, stay <8s |
| S13 | Talking head (opt.) | Host 10s max segments | Between heavy terminal blocks |

**Capture settings (suggested):** 1080p minimum; terminal font 16–18pt; 44.1kHz+ audio; record system + mic on separate tracks if possible.

---

## 5. Beat-by-beat script outline + B-roll

*Tone: calm, instructor-led. Avoid hype adjectives. Use “we” and “you.”*

### Ch. 0 — Cold open (0:00–0:45)

| Beat | Script (talking points) | B-roll / lower-third |
|------|--------------------------|----------------------|
| 0.1 | “If you’re here, you probably already tried **Example 1 — Basic Agent** from the crash course repo.” | Screen: repo README showing “### 1. Basic Agent” |
| 0.2 | “This video is not a faster rerun. It’s the **seams**: the ten ideas the long tutorial *assumes* — so your first hour isn’t guesswork.” | Lower-third: title of this video |
| 0.3 | “By the end you’ll run Basic Agent **and** know what broke when it doesn’t.” | Screen: split second of successful run |

### Ch. 1 — Framing (0:45–2:00)

| Beat | Script | B-roll |
|------|--------|--------|
| 1.1 | “Original course: [YouTube link in description]. Code: [GitHub]. Official docs: [ADK quickstart]. I’m not Google; this is practical orientation.” | Full-screen “links in description” card 3s |
| 1.2 | “We stay on **Example 1** only. No tools, no multi-agent — that’s why this stays beginner-true.” | Highlight Example 1 folder |

### Ch. 2 — Repo map (2:00–4:00)

| Beat | Script | B-roll |
|------|--------|--------|
| 2.1 | “This repo is a **course layout**: many small projects. Each example can have its **own** `.env` — that trips people who set one key at the root and never copy it down.” | Screen: tree; circle example folder |
| 2.2 | “Rule: **whatever folder the README tells you to run from**, treat as the project root for that lesson.” | Annotate arrow to README snippet |

### Ch. 3 — Agent vs script + runtime (4:00–8:00)

| Beat | Script | B-roll |
|------|--------|--------|
| 3.1 | “A **script** calls an API and prints text. An **agent** in ADK is wired into a **runtime**: instructions, model choice, later tools and sessions.” | Diagram S11: left “one-off script” → right “agent + runtime” |
| 3.2 | “If you only copy code without that mental model, every later chapter feels like magic.” | Pause on agent definition file (name varies by repo version — say “open the file that defines your agent” on mic) |
| 3.3 | “Your job today: recognize **where behavior is configured** vs where you type the user question.” | Cursor scroll slow |

### Ch. 4 — venv, pip, Python (8:00–12:00)

| Beat | Script | B-roll |
|------|--------|--------|
| 4.1 | “One venv at repo root is the course design: `python -m venv .venv` then activate **every new terminal**.” | S05, S06 |
| 4.2 | “`pip install -r requirements.txt` — if you skip this, you get import errors that look like ‘ADK is broken.’ It’s not; the environment is.” | Show intentional `ModuleNotFoundError` if safe |
| 4.3 | “Check `python --version`. ADK expects a **modern** Python — if you’re on 3.9 end-of-life, upgrade before you fight ghosts.” | S01 |

### Ch. 5 — `.env`, keys, billing (12:00–16:00)

| Beat | Script | B-roll |
|------|--------|--------|
| 5.1 | “Copy `.env.example` → `.env` **inside the Basic Agent folder** — not optional; the loader looks local.” | S04 |
| 5.2 | “`GOOGLE_API_KEY` — get from AI Studio / Cloud flow per README. Never commit `.env`; add to `.gitignore` if you fork.” | Show `.gitignore` line |
| 5.3 | “Billing in plain English: a key is tied to a **project**. Some flows expect billing enabled even when you think of it as ‘free tier.’ Read Google’s current pricing page before a classroom demo.” | On-screen: “check current Google AI pricing” (no numbers — they change) |

### Ch. 6 — cwd + model IDs (16:00–19:30)

| Beat | Script | B-roll |
|------|--------|--------|
| 6.1 | “Wrong folder = wrong path to config = obscure errors. **cd** into the example folder first; then run the command the README shows.” | S08 |
| 6.2 | “Model ID strings look arbitrary; they’re **not** interchangeable. When a model is renamed or retired, swap the string — don’t assume the tutorial repo is frozen forever.” | Show where model string lives in code |
| 6.3 | “If the model rejects your region or account type, that’s a policy issue, not Python.” | Neutral tone |

### Ch. 7 — Instructions vs user message + sessions teaser (19:30–22:00)

| Beat | Script | B-roll |
|------|--------|--------|
| 7.1 | “**System / developer instructions** shape every reply. The **user message** is the variable part. Mixing them in one blob in your head makes debugging impossible.” | Two-column slide |
| 7.2 | “Even Basic Agent touches **sessions**: the runtime remembers turns so you don’t paste the whole chat back in. Deep state comes in Example 5 in the repo — today, just the idea.” | README crop showing “### 5. Sessions and State” |

### Ch. 8 — Live run + fix one bug (22:00–25:00)

| Beat | Script | B-roll |
|------|--------|--------|
| 8.1 | “Clean run from the correct directory.” | S07 |
| 8.2 | “Now I break it: I `cd` to the wrong place — watch the error.” | S08 — fix live by `cd` correct |
| 8.3 | “Second break: rename `.env` away — see the failure mode.” | S09 — restore file |

### Ch. 9 — Close (25:00–26:30)

| Beat | Script | B-roll |
|------|--------|--------|
| 9.1 | “Next video in *this* series: **Example 2 — Tool Agent** — we add tools and watch the runtime change.” | Repo README “### 2. Tool Agent” |
| 9.2 | **(Optional STC)** | “For CityU STC bridge content, the pattern is the same: **happy path plus honest seams** — that’s how adults trust a technical school.” |

---

## 6. Extended cut add-ons (+8–12 min total)

| Segment | Content |
|---------|---------|
| FAQ rapid fire | “Do I need Vertex?” “Can I use Ollama?” “Windows vs Mac?” — one sentence each, point to docs |
| Homework | “Break three things on purpose; screenshot the error + fix” |
| Instructor notes | Link to [ADK issue discussions](https://github.com/google/adk-python/issues) only as “if you hit a framework bug, search here” — avoid blaming tutorial authors |

---

## 7. Description template (YouTube)

```text
Example #1 from the ADK crash course — the concepts beginners miss: venv, .env per folder, cwd, agent vs script, model IDs, keys/billing, sessions teaser, and live bugfix.

Original course: https://youtu.be/P4VFL9nIaIA
Code repo: https://github.com/imvickykumar/agent-development-kit-crash-course
ADK docs: https://google.github.io/adk-docs/get-started/quickstart

Not affiliated with Google. APIs and pricing change — verify current Google documentation.

Chapters: (paste after upload)
```

---

## 8. Accessibility & trust (align with internal “Video trust” checklist)

- Chapters on every major concept; captions with **correct** model/API names.  
- Disclose if script or B-roll used **generative AI**.  
- Pin a comment: “If Example 1 folder name differs in your fork, drop the path below.”

---

## 9. Revision log

| Date | Change |
|------|--------|
| 2026-02-12 | Initial pack: plan + shot list + script beats + extensions |

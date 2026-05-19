# Feasibility Study: Digital Twins Laboratory
## School of Technology and Computing (STC)
### City University of Seattle
**Prepared by:** Clark Ngo
**Date:** May 2026
**Version:** Draft 3.0

---

## 1. Executive Summary

This feasibility study evaluates the viability of establishing a Digital Twins laboratory within the School of Technology and Computing (STC) at City University of Seattle. The study examines six distinct implementation variations, assessing each against STC's key constraints: limited physical space, budget sensitivity, implementation complexity, and long-term maintainability.

Digital Twins technology — the real-time virtual representation of physical systems — is rapidly becoming a foundational skill across engineering, manufacturing, IoT, and AI industries. Establishing a lab-based curriculum positions STC graduates with a competitive, future-ready skillset aligned with current industry demands.

After evaluating all variations, this study finds that **a phased, cloud-and-simulation-first approach** — beginning with low-cost, software-only tools and scaling toward cloud platforms — offers the most feasible path for STC given its size and resource profile. However, all viable options are documented herein to support informed decision-making by department leadership and the dean.

---

## 2. Introduction

### 2.1 Purpose
This study was commissioned by the School of Technology and Computing at City University of Seattle to assess the feasibility of launching a Digital Twins laboratory for educational purposes. The goal is to equip students with hands-on experience in Digital Twin concepts, tools, and workflows used in real-world industry.

### 2.2 Scope
This document covers:
- A technical overview of Digital Twins and Physical AI
- Educational context and institutional benchmarks
- Six distinct implementation variations
- Comparative analysis across cost, complexity, space, and maintainability
- Recommendations and a phased implementation roadmap

### 2.3 Institutional Context & Constraints
STC operates within a small urban university environment. The following constraints frame all evaluations in this study:

| Constraint | Description |
|---|---|
| **Physical Space** | **No dedicated lab space available.** A small unused server room exists but contains only decade-old, outdated hardware. Cannot be relied upon as primary infrastructure. |
| **Budget** | Preference for free/open-source tools; dean willing to support justified costs |
| **Complexity** | Must be executable by a small faculty team without dedicated DevOps staff |
| **Maintainability** | Tools must be sustainable long-term without significant ongoing effort |
| **Student Audience** | Undergraduate and graduate CS/IT/engineering students — **majority have non-technical or limited programming backgrounds** |

---

## 3. What Are Digital Twins?

A **Digital Twin** is a real-time, dynamic virtual representation of a physical object, process, or system. It continuously ingests data from sensors, IoT devices, or simulated environments to reflect the state of its physical counterpart, enabling monitoring, analysis, prediction, and control.

### 3.1 Core Components
A functional Digital Twin system typically consists of:

- **Physical Layer** — Real or simulated sensors, devices, and actuators
- **Connectivity Layer** — Data pipelines (e.g., MQTT, REST APIs, Node-RED flows)
- **Data Processing Layer** — Cloud platforms or edge computing nodes
- **Digital Model Layer** — 3D models, simulation environments (e.g., NVIDIA Omniverse, Unity)
- **AI/Analytics Layer** — Predictive models, AI agents, anomaly detection
- **Visualization Layer** — Dashboards, 3D renderings, telemetry displays

### 3.2 Physical AI and the Sim-to-Real Pipeline
A rapidly growing subfield is **Physical AI** — training AI models within simulated environments and deploying them to real-world hardware. This "sim-to-real" pipeline is central to modern robotics and autonomous systems development. Key enabling concepts include:

- **Synthetic Data Generation** — Using simulated environments to create large-scale training datasets without physical hardware ([NVIDIA Synthetic Data](https://www.nvidia.com/en-us/use-cases/synthetic-data-generation-for-agentic-ai/))
- **World Foundation Models (WFMs)** — Large AI models (e.g., [NVIDIA Cosmos](https://www.nvidia.com/en-us/omniverse/)) capable of understanding and generating physics-based simulations
- **ROS 2 (Robot Operating System)** — The industry-standard middleware for robotics communication and control ([ros.org](https://docs.ros.org/en/humble/index.html))
- **OpenUSD (Universal Scene Description)** — A framework for describing, composing, and simulating 3D environments ([NVIDIA OpenUSD Docs](https://docs.nvidia.com/learn-openusd/latest/index.html))

### 3.3 Teaching Tracks
Based on industry trends and available tooling, two distinct educational tracks emerge:

| Track | Focus | Key Tools |
|---|---|---|
| **Track A: Robotics & Simulation** | Sim-to-real pipeline, autonomous robots, Physical AI | NVIDIA Isaac Sim, ROS 2, Wokwi, OpenUSD |
| **Track B: Industrial Data Analytics** | IoT data flows, industrial monitoring, cloud integration | Azure Digital Twins, Node-RED, XMPro |

Both tracks are evaluated across all implementation variations in Section 6.

---

## 3.5 Why Digital Twins and Physical AI Matter Right Now — A Realistic Assessment

It is worth being direct: Digital Twins are not new. The concept has existed since NASA used simulation models for space missions in the 1960s. What *is* new — and what makes this the right moment for an institution like STC to act — is that the technology has finally become accessible, affordable, and industry-standard at the same time that the workforce gap is widening faster than universities can respond.

### 3.5.1 The Convergence That Changed Everything

For most of its history, Digital Twin technology required expensive proprietary software, massive hardware infrastructure, and teams of specialized engineers. That era ended around 2022–2024, driven by three simultaneous shifts:

**1. Cloud commoditization.** Azure, AWS, and Google Cloud now offer managed IoT and Digital Twin services at consumption-based pricing — a student can build a functional twin for under $10. What previously required an enterprise infrastructure budget is now accessible from a laptop.

**2. Open-source maturation.** Tools like Eclipse Ditto, Node-RED, and ROS 2 have reached production-grade stability. NVIDIA released Omniverse as free for individuals. AMD open-sourced ROCm. The Genesis simulator runs on a laptop. The "enterprise tax" on simulation technology has effectively been eliminated for education.

**3. AI integration.** The addition of AI — particularly generative models, reinforcement learning, and World Foundation Models — transformed Digital Twins from passive monitoring dashboards into active, predictive, and autonomous systems. This is what the industry now calls **Physical AI**: systems that understand and operate in the physical world by learning in simulation first.

These three shifts happened within the same five-year window. The result is a technology that is simultaneously more powerful, more affordable, and more in-demand than at any prior point in its history.

### 3.5.2 Market Reality — Not Projection, But Present

The numbers are not forecasts — they describe what is happening now:

- The global Digital Twin market was valued at **~$28–35 billion in 2025** and is growing at a **CAGR of 31–47%** depending on the segment, with North America holding the largest share at ~34% ([Fortune Business Insights](https://www.fortunebusinessinsights.com/digital-twin-market-106246), [Grand View Research](https://www.grandviewresearch.com/industry-analysis/digital-twin-market))
- The U.S. Digital Twin market alone is projected to grow from **$3.9 billion in 2025 to $29.8 billion by 2032** ([Fortune Business Insights](https://www.fortunebusinessinsights.com/u-s-digital-twin-market-107449))
- **72% of manufacturers** plan to deploy Digital Twin technology in their operations by 2026 ([Intellect Markets](https://www.intellectmarkets.com/report/digital-twin-market))
- **80%+ of enterprise-level deployments** now combine Digital Twins with AI and IoT ([Intellect Markets](https://www.intellectmarkets.com/report/digital-twin-market))
- McKinsey projects the market could grow ~60% annually to reach **$73.5 billion by 2027**, describing 2026 as a *"scaling moment"* ([StartUs Insights](https://www.startus-insights.com/innovators-guide/digital-twin-report/))
- Siemens acquired Altair Engineering for **$10 billion in March 2025** specifically to strengthen its Digital Twin and industrial AI capabilities ([MarketsandMarkets](https://www.marketsandmarkets.com/Market-Reports/digital-twin-market-225269522.html))

This is not a speculative technology. It is infrastructure that is actively being deployed at scale across manufacturing, aerospace, healthcare, smart cities, and energy.

### 3.5.3 The Workforce Gap Is Real and Growing

Here is the honest challenge: universities have not caught up. The same market reports that document explosive growth also consistently flag a **shortage of qualified workers** as a primary constraint on adoption. This is not abstract.

On LinkedIn as of May 2026, there are **933+ active Digital Twin job postings in the United States** ([LinkedIn Jobs](https://www.linkedin.com/jobs/digital-twin-jobs)). On Indeed, **2,367+ Digital Twin roles** are listed nationally ([Indeed](https://www.indeed.com/q-digital-twin-jobs.html)). Average salaries for Digital Twin roles range from **$98,000–$152,000+** depending on seniority, with senior engineers and architects commanding **$180,000–$280,000** ([ZipRecruiter](https://www.ziprecruiter.com/Salaries/Digital-Twin-Salary), [Glassdoor](https://www.glassdoor.com/Salaries/digital-twin-engineer-salary-SRCH_KO0,21.htm)).

The skills being demanded — IoT integration, cloud platforms, simulation, AI/ML, data pipelines — are exactly what a Digital Twins lab teaches. There is no mismatch between what the lab would deliver and what employers are hiring for.

### 3.5.4 Why This Specific Moment — Not Later

The honest case for acting now rather than waiting is this:

**The early-mover window is closing.** CMU, University of Michigan, and MIT are already running Digital Twins programs. NVIDIA, Microsoft, and AMD have all launched free academic programs specifically because they need university partners to train the next wave of talent. These programs have finite capacity and preferential terms for early adopters. Waiting another two years means competing with institutions that are already established.

**The tools are free right now.** NVIDIA Omniverse, ROCm, Genesis, Eclipse Ditto, Node-RED, and Wokwi are all free or have free academic tiers today. Vendor strategies change. Azure free tiers get restructured. Teaching kits get updated and availability shifts. The current landscape of zero-cost entry is the most favorable it has been, and there is no guarantee it stays this way.

**Students are asking for it already.** The industry signals are visible enough that students researching career paths are encountering Digital Twins, Physical AI, and IoT as recurring requirements in job postings. Institutions that can say "we teach this" have a real enrollment differentiator. Institutions that cannot are quietly losing prospective students to schools that can.

The case is not that Digital Twins are the future. The case is that they are the present — and STC's curriculum does not yet reflect that.

---

## 4. Why a Digital Twins Lab?

### 4.1 Industry Relevance
Digital Twins are no longer experimental — they are production-grade infrastructure across multiple sectors:
- **Manufacturing:** Predictive maintenance, process optimization
- **Healthcare:** Patient monitoring, surgical simulation
- **Smart Cities:** Infrastructure management, traffic systems
- **Aerospace & Defense:** Simulation-based testing, mission planning
- **Retail & Logistics:** Supply chain visibility, warehouse automation

### 4.2 Student Audience Profile
STC's student body spans undergraduate and graduate levels across CS, IT, and engineering programs. However, **the majority enter with limited or no programming experience** — many come from career-change or interdisciplinary backgrounds. This is a critical design constraint for the lab: tools must be approachable without assuming prior command-line, Linux, or coding proficiency. This does not lower the bar for outcomes, but it does shape *how* students are onboarded.

> **Implication for tool selection:** Variations that rely heavily on Linux terminal work, manual environment configuration, or low-level coding (e.g., NVIDIA's full local stack, AMD ROCm setup) carry higher friction for this audience and require more instructor scaffolding. Browser-based, visual, or GUI-first tools (Node-RED, Wokwi, Azure Portal, XMPro) are significantly more accessible as entry points.

### 4.3 Student Outcomes
A Digital Twins lab enables STC students to develop the following competencies, organized by difficulty tier:

**Foundational (accessible to all students):**
- Understanding Digital Twin concepts and architecture
- IoT sensor simulation and data visualization (using browser-based tools)
- Flow-based programming without code (Node-RED drag-and-drop)
- Reading and interpreting real-time dashboards (Grafana, Azure Portal)
- Cloud platform orientation (Azure Portal GUI)

**Intermediate (for students with some technical exposure):**
- IoT data pipelines and MQTT messaging
- Cloud platform integration and basic scripting (Azure, Python)
- Digital twin state modeling (DTDL, Eclipse Ditto)
- 3D scene navigation and asset configuration (Unity, Omniverse GUI)

**Advanced (for technically stronger students):**
- AI model training using synthetic and real data (Isaac Lab, Genesis)
- Robotics programming and simulation (ROS 2, Isaac Sim, Genesis)
- GPU-accelerated computing (ROCm, CUDA)
- Sim-to-real transfer and deployment

> **Recommended approach for mixed-ability cohorts:** Begin all students on foundational tools (V3: Open-Source Stack or V2: Azure) and allow advanced students to progress into simulation tracks (V1: NVIDIA or V6: AMD) as elective or capstone work.

### 4.3 Institutional Benchmarks
Leading academic institutions have already established Digital Twins curricula, validating STC's direction:

| Institution | Program | Approach |
|---|---|---|
| **Carnegie Mellon University** | [AI Engineering Graduate Certificate (Digital Twins & Analytics)](https://www.cmu.edu/) | Mathematical/statistical foundations, predictive analytics, AI ethics |
| **University of Michigan (Coursera)** | [Digital Twins Specialization](https://www.coursera.org/specializations/digital-manufacturing-design-technology) | Manufacturing ecosystem focus, business value and implementation strategy |
| **MIT Professional Education** | [Digital Twins – The Vision Demystified](https://professional.mit.edu/) | Digital thread, sensor-to-computation data flow |
| **NVIDIA Deep Learning Institute** | [Physical AI Learning Path](https://www.nvidia.com/en-us/training/) | Hands-on sim-to-real, Isaac Sim, ROS 2, Cosmos WFMs |

STC has an opportunity to differentiate by offering **hands-on lab-based instruction** at an accessible price point — something the above certifications do not fully provide.

---

## 4.6 Job Market Alignment — Who Is Hiring and Where

### 4.6.1 Salary Landscape (United States, 2025–2026)

Digital Twin and Physical AI roles command strong compensation at every level, reflecting genuine market scarcity:

| Role | Entry Level | Mid-Level | Senior | Source |
|---|---|---|---|---|
| Digital Twin Developer | $85,000–$110,000 | $110,000–$145,000 | $145,000–$180,000 | [SecondTalent, 2026](https://www.secondtalent.com/occupations/digital-twin-developer/) |
| Digital Twin Engineer | $113,980–$136,000 | $136,000–$151,973 | $180,000–$278,110 | [Glassdoor, Apr 2026](https://www.glassdoor.com/Salaries/digital-twin-engineer-salary-SRCH_KO0,21.htm) |
| Digital Twin Architect / Lead | $180,000–$220,000 | $220,000–$280,000+ | — | [SecondTalent, 2026](https://www.secondtalent.com/occupations/digital-twin-developer/) |
| AI/ML Engineer (Seattle) | — | ~$140,000–$206,000 | $200,000–$312,000 | [ZipRecruiter](https://www.ziprecruiter.com/Jobs/Ai/-in-Seattle,WA), [SecondTalent](https://www.secondtalent.com/resources/most-in-demand-ai-engineering-skills-and-salary-ranges/) |
| IoT / Systems Integration Engineer | $80,000–$110,000 | $110,000–$145,000 | $145,000–$180,000+ | Industry average |

> The IT industry average salary in greater Seattle is **$361,802** according to Lightcast (2025) — reflecting the concentration of high-compensation roles at Amazon, Microsoft, and adjacent tech companies ([Greater Seattle IT](https://greater-seattle.com/it/)).

### 4.6.2 Employers in Seattle and the Puget Sound Region

Seattle is one of the most concentrated technology labor markets in the world. The following employers are actively hiring in Digital Twins, IoT, simulation, robotics, and Physical AI — the exact skill domains a Digital Twins lab teaches:

| Employer | Relevance to Digital Twins / Physical AI | Location |
|---|---|---|
| **Microsoft** | Azure Digital Twins (the platform itself), Azure IoT Hub, HoloLens mixed reality, AI Copilot infrastructure | Redmond / Seattle |
| **Amazon / Amazon Robotics** | Warehouse automation, Physical AI for fulfillment robots, digital twin of logistics systems; AWS IoT TwinMaker | Seattle / Bellevue |
| **Boeing** | Digital twin of aircraft components, manufacturing simulation, aerospace systems modeling | Renton / Everett |
| **NVIDIA** (Seattle office) | Isaac Sim, Omniverse, robotics AI — NVIDIA has a growing presence in the Seattle area | Seattle |
| **Google / DeepMind** (Seattle) | AI infrastructure, simulation environments, robotics research | Kirkland / Seattle |
| **Meta / Reality Labs** | AR/VR and simulation platforms, digital twin for physical environments | Bellevue |
| **Carbon Robotics** | AI-powered agricultural robotics, computer vision, real-time simulation | Seattle |
| **Convoy / Flexport** | Supply chain digital twins, logistics data modeling | Seattle |
| **Accenture** (Seattle office) | Enterprise Digital Twin consulting and implementation | Seattle |
| **T-Mobile** | Network digital twins for 5G infrastructure management | Bellevue |
| **Starbucks** | Digital twin of supply chain and store operations (emerging) | Seattle |
| **University of Washington** | Research partnerships: robotics, AI systems, simulation labs | Seattle |

> Seattle added **29,336 more tech jobs than tech degrees earned** over the prior five years — meaning the region chronically runs a talent deficit ([Greater Seattle IT](https://greater-seattle.com/it/)). STC graduates directly address this gap.

### 4.6.3 Employers in Washington State (Beyond Seattle)

| Employer | Relevance | Location |
|---|---|---|
| **Boeing Defense** | Advanced aircraft simulation, digital manufacturing | Everett, Renton |
| **PACCAR** (Kenworth/Peterbilt) | Vehicle simulation, autonomous truck development | Bellevue |
| **Puget Sound Energy** | Energy grid digital twins, predictive infrastructure monitoring | Bellevue |
| **Joint Base Lewis-McChord (JBLM)** | Defense simulation, training systems | Tacoma |
| **Schweitzer Engineering Labs** | Industrial IoT, power grid monitoring | Pullman |
| **WSU / Pacific Northwest National Lab** | Research: energy systems, digital twin for scientific infrastructure | Richland / Pullman |
| **Port of Seattle / Port of Tacoma** | Smart port infrastructure, logistics digital twins | Seattle / Tacoma |

### 4.6.4 National Employers — Key Demand Hubs

Beyond the Pacific Northwest, Digital Twin talent is in demand across the country's major industrial and technology centers:

| Region | Key Employers | Industries |
|---|---|---|
| **San Francisco / Silicon Valley** | NVIDIA (HQ), Google, Apple, Tesla, Autodesk, Palantir | Robotics, AI, autonomous vehicles, simulation software |
| **Detroit / Midwest** | Ford, GM, GE, Siemens USA, Rockwell Automation | Automotive, manufacturing, industrial IoT |
| **Houston / Texas** | Chevron, ExxonMobil, Schlumberger, Dell | Energy, oil & gas digital twins, industrial analytics |
| **New York / Boston** | Accenture, IBM, GE Digital, PTC, Bentley Systems | Enterprise consulting, aerospace, industrial software |
| **Raleigh / Atlanta** | GE Aerospace, Honeywell, Siemens | Aerospace, defense, smart manufacturing |
| **Remote / distributed** | Microsoft, Amazon, IBM, Accenture (all have distributed hiring) | All sectors |

> According to Indeed, **2,367+ Digital Twin jobs** are currently posted nationally, with concentration in tech hubs but also in aerospace, energy, and manufacturing clusters across the Midwest and South ([Indeed Digital Twin Jobs](https://www.indeed.com/q-digital-twin-jobs.html)).

### 4.6.5 Job Titles STC Graduates Could Target

An STC student completing a Digital Twins lab curriculum is positioned for the following roles at graduation or within 1–2 years:

**Entry-level (0–2 years experience):**
- IoT Engineer / IoT Developer
- Digital Twin Analyst
- Junior Simulation Engineer
- Cloud Infrastructure Associate (Azure IoT)
- Data Pipeline Engineer

**Mid-level (2–5 years):**
- Digital Twin Developer / Engineer
- Robotics Software Engineer (with simulation background)
- AI Systems Engineer
- Industrial IoT Architect
- Platform Engineer (Azure / AWS IoT)

**Advanced / Specialist:**
- Digital Twin Architect
- Physical AI Engineer
- Simulation & Validation Lead
- Director of Digital Engineering

> These roles are not speculative. They appear verbatim in active job postings on LinkedIn, Indeed, and ZipRecruiter as of May 2026.

---

## 4.5 Physical Infrastructure Assessment

### 4.5.1 Current State
STC currently has:
- **No dedicated student lab space** for a Digital Twins lab
- **One small, unused server room** containing hardware that is approximately a decade old
- **No confirmed working components** — all existing hardware should be treated as untested until physically inspected

This is a significant constraint that rules out any implementation variation requiring dedicated physical workstations, hardware peripherals, or on-premise servers as primary infrastructure. **The lab must be designed around what students already have (laptops) and what STC can access remotely (cloud).**

### 4.5.2 Server Room Hardware: Repurposability Assessment

The existing server room hardware may still have limited value as a secondary, non-critical backend host — but only after a physical inspection confirms minimum viable specs.

**Minimum specs required to host the open-source backend stack (Eclipse Ditto + Node-RED + InfluxDB + Grafana + Mosquitto via Docker Compose):**

| Component | Minimum Viable | Typical Decade-Old Server | Verdict |
|---|---|---|---|
| CPU | 4-core, 64-bit (2010+) | Intel Xeon E5 / Core i5-i7 (2012–2015 era) | ✅ Likely sufficient for lightweight backend only |
| RAM | 8 GB | 8–16 GB DDR3 (may need upgrade) | ⚠️ Check; may need RAM upgrade (~$20–$40) |
| Storage | 50 GB free, ideally SSD | HDD likely (slow); SSD unlikely | ⚠️ HDD degrades performance; SSD upgrade recommended (~$50–$80) |
| **GPU** | **None required for V3 backend** | **❌ Confirmed: no graphics card present** | **✅ Not needed for open-source stack. Rules out all simulation variations (V1, V4, V6) entirely on this hardware.** |
| OS compatibility | Ubuntu 22.04 LTS | 64-bit Intel/AMD hardware | ✅ Compatible |
| Network | Stable LAN/internet | Existing campus network | ✅ Likely fine |
| Cooling | Working server room cooling | Must verify | ⚠️ Must confirm |
| Power | Stable power supply | Unknown condition | ⚠️ UPS recommended |

**Key risks of relying on decade-old hardware:**

| Risk | Severity | Notes |
|---|---|---|
| Unexpected hardware failure mid-semester | High | HDDs and capacitors on 10-year-old hardware fail without warning |
| No support or replacement parts readily available | High | Proprietary server hardware from 2013–2015 may have limited parts availability |
| High power consumption | Medium | Old servers are energy-inefficient; ongoing electricity cost adds up |
| Security vulnerabilities | Medium | Old hardware may not support firmware/BIOS updates needed for modern OS security |
| Student access limitations | Medium | Server room is not a student workspace — remote access only |

### 4.5.3 Recommendation: Cloud-First, Old Server as Optional Backup Only

The existing server room has **no GPU**. This is a hard constraint that determines exactly what it can and cannot do:

| Capability | Old Server Room | Why |
|---|---|---|
| Host V3 open-source backend (Docker, Ditto, Node-RED, Grafana) | ✅ Possibly — after inspection | CPU/RAM only required; no GPU needed |
| Run NVIDIA Isaac Sim or Omniverse (V1) | ❌ Not possible | Requires NVIDIA GPU |
| Run Genesis / ROCm (V6 AMD) | ❌ Not possible | Requires AMD GPU |
| Run Unity simulations (V4) | ❌ Not possible | Requires GPU for rendering |
| Serve Azure Digital Twins portal (V2) | ✅ Irrelevant — browser-based | Azure runs in the cloud; old hardware has no role |
| Replace a cloud VM entirely | ⚠️ Risky — not recommended | Decade-old hardware has high failure risk |

**The old server room hardware can only ever serve one purpose in this plan: optionally hosting the V3 open-source backend services.** Everything else — including all simulation tools — requires cloud access or new hardware.

Given the failure risks, STC should not depend on this hardware for student-facing services. A $10–$20/month cloud VM is the strongly preferred primary option.

### 4.5.4 What STC Actually Needs to Start (Physical Checklist)

Since there is no lab space and no usable hardware, the following is the **complete physical infrastructure STC needs to launch Phase 1:**

| Item | What's Needed | Where to Get It | Cost |
|---|---|---|---|
| Student access device | Any modern laptop (2018+) or Chromebook | Students' own devices | $0 |
| Internet connection | Reliable campus WiFi or wired | Existing campus network | $0 |
| Cloud VM for backend services | 2 vCPU, 4–8 GB RAM, 50 GB SSD | DigitalOcean, Linode, or AWS Lightsail | ~$10–$20/month |
| Faculty management laptop | Any modern laptop | Existing | $0 |
| **Total Phase 1 physical requirement** | | | **~$10–$20/month** |

No lab space, no dedicated hardware, no server room required to begin.

---

## 5. Constraints & Requirements Summary

Before evaluating variations, the following requirements matrix defines what "feasible" means for STC:

| Requirement | Priority | Threshold |
|---|---|---|
| Low upfront cost | High | Prefer free; < $5,000 acceptable for dean-approved setup |
| No dedicated lab or server space | **Critical** | **No student lab space exists. All tools must run on student laptops and/or cloud. No variation requiring physical lab workstations is viable as primary infrastructure.** |
| Low implementation complexity | High | Setup completable by 1–2 faculty members |
| Low ongoing maintenance burden | High | No dedicated sysadmin required |
| **Accessible to non-technical students** | **High** | **Tools must have GUI, browser-based, or visual interfaces as the entry point; command-line should be optional or scaffolded** |
| Scalable to more students | Medium | Should support 15–30 concurrent students |
| Curriculum-ready resources | Medium | Existing teaching kits, tutorials, or course materials preferred |
| Industry-recognized tools | Medium | Tools used in real-world practice |

---

## 6. Implementation Variations

---

### Variation 1: NVIDIA Omniverse Ecosystem (Simulation-First)

#### Overview
This variation builds the lab entirely on NVIDIA's simulation and Physical AI stack, leveraging free academic licenses and pre-built teaching kits. It is oriented toward **Track A (Robotics & Simulation)**.

#### Core Tools & Links

| Tool | Role | Cost | Link |
|---|---|---|---|
| NVIDIA Omniverse | 3D simulation platform and Digital Twin environment | Free (individual/academic, up to 2 collaborators) | [omniverse.nvidia.com](https://www.nvidia.com/en-us/omniverse/) |
| NVIDIA Isaac Sim | Robotics simulation built on Omniverse | Free with Omniverse | [Isaac Sim Docs](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) |
| NVIDIA Isaac Lab | Reinforcement learning and robot policy training | Free, open-source | [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab) |
| ROS 2 | Robotics middleware (perception, navigation, control) | Free, open-source | [docs.ros.org](https://docs.ros.org/en/humble/index.html) |
| OpenUSD | 3D scene description and asset interoperability | Free, open-source | [OpenUSD Docs](https://docs.nvidia.com/learn-openusd/latest/index.html) |
| NVIDIA Cosmos | World Foundation Model for synthetic data generation | API-based; free tier via NVIDIA NGC | [Cosmos](https://www.nvidia.com/en-us/use-cases/synthetic-data-generation-for-agentic-ai/) |
| NVIDIA Teaching Kits | Edge AI, Robotics, Generative AI lab kits | Free for faculty | [DLI Teaching Kits](https://www.nvidia.com/en-us/training/educator-programs/teaching-kits/) |
| NVIDIA Sim-to-Real Course | SO-101 Sim-to-Real transfer course | Free | [Sim-to-Real SO-101](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/01-overview.html) |

> **Important licensing note:** NVIDIA Omniverse is free for individuals and academic use with up to 2 collaborators on shared project content. If 3 or more faculty/students collaborate on the same project asset, Omniverse Enterprise Subscription Licenses are required. Academic pricing is available through the [NVIDIA Education Pricing Program](https://developer.nvidia.com/education-pricing). For classroom use where students work on individual projects, the free tier applies.

#### What Students Learn
- Building and assembling industrial 3D scenes in Omniverse
- Generating synthetic training datasets using Cosmos
- Training robot policies in Isaac Lab and deploying via sim-to-real transfer
- ROS 2-based robot perception and navigation
- Vision-language model integration for factory intelligence
- OpenUSD scene composition and asset interoperability

#### Hardware Requirements & Department Cost Estimate

> ⚠️ **Space constraint reminder:** STC has no student lab space. Any on-premise workstation options below are future investment scenarios only — **not viable for initial deployment.** Cloud-first is the only immediately feasible path.

This is the most hardware-sensitive variation. Isaac Sim requires a dedicated NVIDIA GPU to run locally.

**Option A: Cloud-First via NVIDIA Omniverse Cloud (Recommended — Zero Hardware, Zero Lab Space)**
Students access the full simulation environment through a browser. No GPU, no lab space, no installation required.

| Item | Quantity | Unit Cost | Total |
|---|---|---|---|
| Omniverse Cloud access | Per-user (pricing via NVIDIA sales — contact for academic rates) | TBD | Contact NVIDIA |
| Student laptops (existing) | 0 new purchases | — | $0 |
| **Total (Cloud, immediate)** | | | **~$0–$TBD** |

> Contact NVIDIA academic sales at [developer.nvidia.com/education-pricing](https://developer.nvidia.com/education-pricing) for Omniverse Cloud academic rates.

**Option B: On-Premise Lab Workstations (Future Investment — Requires Lab Space STC Does Not Currently Have)**
This option is only viable if STC acquires dedicated lab or classroom space in the future.

| Item | Quantity | Unit Cost | Total |
|---|---|---|---|
| GPU Workstation (RTX 4080, education-grade) | 3–5 units | ~$2,500–$4,500 | ~$7,500–$22,500 |
| Monitor (27", 4K) | 3–5 units | ~$300–$500 | ~$900–$2,500 |
| Setup & IT configuration | 1-time | ~$500 | ~$500 |
| **Total (On-Premise, future)** | | | **~$8,900–$25,500** |

**Option C: Hybrid — 2 Faculty GPU Workstations (Minimum Viable If Space Is Found)**
Two workstations placed in any available classroom or office space serve as shared simulation nodes; students remote-connect or take turns.

| Item | Quantity | Unit Cost | Total |
|---|---|---|---|
| GPU Workstation (RTX 4080) | 2 units | ~$2,800 | ~$5,600 |
| Monitor | 2 units | ~$400 | ~$800 |
| **Total (Hybrid, future)** | | | **~$6,400** |

> **Old server room hardware note:** STC's existing server room has **no graphics card**. It cannot run NVIDIA Isaac Sim, Omniverse, or any GPU-accelerated simulation workload under any circumstances. Cloud access (Option A) is the only viable path for this variation given current infrastructure.

#### Implementation Steps (Detailed)

**Step 1: Register for NVIDIA DLI Faculty Program**
- Visit [NVIDIA Teaching Kits](https://www.nvidia.com/en-us/training/educator-programs/teaching-kits/)
- Register as faculty to receive free access to the Edge AI & Robotics Teaching Kit and Generative AI Teaching Kit
- Download pre-built lab exercises, datasets, and instructor guides

**Step 2: Set Up the Omniverse Environment**
- If using cloud: Create an NVIDIA Developer account at [developer.nvidia.com](https://developer.nvidia.com/)
- If local: Install Omniverse via NGC Catalog (Omniverse Launcher was deprecated October 2025; use NGC instead)
- Install Isaac Sim as an Omniverse extension
- Install ROS 2 Humble on lab machines (Ubuntu 22.04 recommended)
- Configure the Isaac ROS bridge to connect ROS 2 with Isaac Sim

**Step 3: Set Up Isaac Lab for Robot Training**
- Clone the [Isaac Lab GitHub repository](https://github.com/isaac-sim/IsaacLab)
- Follow the installation guide to configure the reinforcement learning environment
- Configure sample robot environments (Franka arm, quadruped, humanoid options available)

**Step 4: Design Curriculum Around DLI Teaching Kits**
- Module 1: Introduction to OpenUSD and 3D scene assembly (weeks 1–2)
- Module 2: Synthetic data generation with Cosmos (weeks 3–4)
- Module 3: Robot policy training in Isaac Lab (weeks 5–7)
- Module 4: Sim-to-Real transfer and ROS 2 deployment (weeks 8–10)
- Module 5: Capstone — build and simulate a complete industrial digital twin (weeks 11–14)

**Step 5: Deliver via OpenEdX**
- Set up [OpenEdX](https://openedx.org/) as the LMS (self-hosted on a $10–20/month cloud VM)
- Upload lab assignments, video walkthroughs, and grading rubrics
- Embed Omniverse Cloud links or provide local connection instructions

#### Feasibility Assessment

| Factor | Rating | Notes |
|---|---|---|
| Cost | ⭐⭐⭐⭐⭐ | Software free; cost limited to optional GPU hardware |
| Implementation Complexity | ⭐⭐ | High barrier for non-technical students: requires Linux, terminal setup, Python, and GPU configuration. Suitable as an advanced/elective track — not recommended as a first course for non-tech students. Faculty scaffolding is essential. |
| Space Requirements | ⭐⭐⭐⭐⭐ | Cloud option requires zero physical footprint |
| Maintainability | ⭐⭐⭐⭐ | NVIDIA maintains software; regular updates and solid support |
| Curriculum Readiness | ⭐⭐⭐⭐⭐ | Extensive free teaching kits, DLI courses, documentation |
| Industry Relevance | ⭐⭐⭐⭐⭐ | NVIDIA is the dominant platform in robotics and Physical AI |

**Verdict:** Strongest option for a forward-looking, simulation-focused curriculum — but **not suitable as a first course for non-technical students**. The command-line setup, GPU configuration, and Python/ROS 2 requirements present a significant barrier without prior tech experience. Recommended as a **second-year elective or advanced capstone track**, after students have built foundational skills through V3 or V2. Cloud option eliminates space and hardware barriers; student accessibility remains the primary challenge.

#### Getting Started Resources

| Resource | Type | Link |
|---|---|---|
| Isaac Sim Quickstart Index | Official getting started | [docs.isaacsim.omniverse.nvidia.com – Quickstart](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/introduction/quickstart_index.html) |
| Isaac Sim Basic Usage Tutorial | Step-by-step (GUI + Python) | [Quickstart: Isaac Sim Basic Usage](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/introduction/quickstart_isaacsim.html) |
| Building Your First Robot in Isaac Sim | Beginner learning path | [NVIDIA Learning: Getting Started with Isaac Sim](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/building-your-first-robot-in-isaac-sim/index.html) |
| Getting Started with Isaac Sim (Full Path) | Comprehensive modules | [NVIDIA Learning: Isaac Sim Full Path](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/) |
| Sim-to-Real SO-101 Course | Sim-to-real transfer | [NVIDIA SO-101 Overview](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/01-overview.html) |
| Isaac ROS Getting Started | ROS 2 integration | [Isaac ROS Getting Started](https://nvidia-isaac-ros.github.io/getting_started/index.html) |
| NVIDIA DLI Teaching Kits (Faculty) | Free course materials | [NVIDIA Teaching Kits](https://www.nvidia.com/en-us/training/educator-programs/teaching-kits/) |
| Edge AI & Robotics Teaching Kit | Lab exercises (Bitbucket) | [Edge AI & Robotics Kit](https://bitbucket.org/nvidia-dli/edge-ai-and-robotics-teaching-kit-labs/src/master/) |
| OpenUSD Getting Started | 3D scene description | [OpenUSD Docs](https://docs.nvidia.com/learn-openusd/latest/index.html) |

#### Example End Goal: Autonomous Warehouse Robot Digital Twin
A student team builds a **simulated autonomous warehouse robot arm** in NVIDIA Isaac Sim using the Franka Emika Panda robot asset. The robot is trained in Isaac Lab using reinforcement learning to pick and place boxes on a conveyor belt. Synthetic training data is generated using Cosmos. The trained policy is tested via sim-to-real transfer in a ROS 2 environment. The final deliverable is a 3D Omniverse scene showing the robot operating in real time with telemetry overlaid — a digital twin of a smart warehouse cell.

**Key skills demonstrated:** OpenUSD scene assembly, robot physics simulation, synthetic data generation, RL policy training, ROS 2 integration, sim-to-real transfer.

---

### Variation 2: Azure Digital Twins (Cloud-Based Industrial)

#### Overview
This variation uses Microsoft Azure's managed Digital Twins service, oriented toward **Track B (Industrial Data Analytics)**. It mirrors the approach described in *Building Industrial Digital Twins* (Nath, 2021) and is well-suited for IoT, data engineering, and cloud architecture courses.

#### Core Tools & Links

| Tool | Role | Cost | Link |
|---|---|---|---|
| Azure Digital Twins | Core platform for modeling digital twin graphs | Pay-per-use: $1/million ops, $2.50/million messages, $0.50/million query units | [Azure Digital Twins Pricing](https://azure.microsoft.com/en-us/pricing/details/digital-twins/) |
| Azure IoT Hub | Device connectivity and telemetry ingestion | Free tier: 8,000 msgs/day; S1 tier: ~$25/month | [Azure IoT Hub](https://azure.microsoft.com/en-us/products/iot-hub/) |
| Azure Time Series Insights / ADX | Time-series data analytics | Pay-per-use | [Azure ADX](https://azure.microsoft.com/en-us/products/data-explorer/) |
| Node-RED | Visual IoT data flow programming | Free, open-source | [nodered.org](https://nodered.org/) |
| Wokwi | Browser-based hardware simulation (Arduino, ESP32) | Free | [wokwi.com](https://wokwi.com/) |
| Azure for Students | $100 credit + free services | Free — no credit card required | [Azure for Students](https://azure.microsoft.com/en-us/free/students/) |
| Microsoft Azure Digital Twins Explorer | Browser-based twin graph visualization tool | Free (developer tool) | [ADT Explorer](https://learn.microsoft.com/en-us/azure/digital-twins/concepts-azure-digital-twins-explorer) |

#### What Students Learn
- Modeling physical environments as digital twin graphs using [DTDL (Digital Twins Definition Language)](https://learn.microsoft.com/en-us/azure/digital-twins/concepts-models)
- Connecting simulated IoT devices (via Wokwi) to Azure IoT Hub
- Building real-time telemetry pipelines using Node-RED
- Querying and visualizing live digital twin state in Azure Digital Twins Explorer
- Event-driven architecture with Azure Event Grid

#### Hardware Requirements & Department Cost Estimate

This is the lightest option hardware-wise — everything runs in the cloud.

**Department Infrastructure Cost (No new hardware needed)**

| Item | Quantity | Unit Cost / Month | Annual Cost |
|---|---|---|---|
| Azure IoT Hub (S1 tier, shared classroom instance) | 1 instance | ~$25/month | ~$300/year |
| Azure Digital Twins operations (classroom usage — approx. 10M ops/year) | — | ~$10/month | ~$120/year |
| Azure Data Explorer (optional, for analytics labs) | 1 cluster | ~$140/month | ~$1,680/year |
| OpenEdX VM (course delivery) | 1 VM (2 vCPU) | ~$15/month | ~$180/year |
| **Total Annual Operating Cost** | | | **~$430–$2,280/year** |

> **Cost mitigation:** Each student enrolled in the [Azure for Students](https://azure.microsoft.com/en-us/free/students/) program receives $100 in free Azure credits (no credit card required). With 20 students, that's $2,000 in free cloud credits per cohort — enough to cover all lab exercises if structured correctly. The department only pays for shared infrastructure (IoT Hub, OpenEdX VM).

> Additionally, STC should investigate the [Microsoft Azure Academic Alliance / Microsoft Imagine](https://azure.microsoft.com/en-us/free/students/) program, which may provide institutional Azure credits beyond the per-student allocation.

**On-Premise Hardware (Optional — Low Priority)**
No dedicated hardware is required. Wokwi simulates all IoT hardware in the browser. If STC later wants physical IoT kits for hands-on labs:

| Item | Quantity | Unit Cost | Total |
|---|---|---|---|
| ESP32 DevKit starter kit (per student group) | 5–8 kits | ~$15–$25 | ~$75–$200 |
| USB cables, breadboards, sensors bundle | 5–8 sets | ~$10–$20 | ~$50–$160 |
| **Optional Physical IoT Kit Total** | | | **~$125–$360** |

#### Implementation Steps (Detailed)

**Step 1: Set Up Microsoft Academic Access**
- Register STC with [Microsoft Azure Dev Tools for Teaching](https://azureforeducation.microsoft.com/devtools)
- Enroll students in [Azure for Students](https://azure.microsoft.com/en-us/free/students/) for $100 free credits each
- Access [Microsoft Learn](https://learn.microsoft.com/en-us/azure/digital-twins/) for free Azure Digital Twins training modules

**Step 2: Provision the Shared Classroom Azure Environment**
- Create an Azure subscription under the department account
- Deploy one Azure Digital Twins instance for instructor use
- Deploy one Azure IoT Hub (S1 tier) for device telemetry ingestion
- Set up Azure Digital Twins Explorer for visual graph editing

**Step 3: Configure Node-RED for Data Flow Exercises**
- Install [Node-RED](https://nodered.org/docs/getting-started/) on a shared cloud VM (or locally on student laptops)
- Install the [Node-RED Azure IoT Hub connector](https://flows.nodered.org/node/node-red-contrib-azure-iot-hub)
- Build sample flows: simulated sensor → IoT Hub → Azure Digital Twins graph update → dashboard

**Step 4: Integrate Wokwi for Hardware Simulation**
- Direct students to [wokwi.com](https://wokwi.com/) — no installation required
- Use pre-built Wokwi projects simulating temperature sensors, motion detectors, and actuators
- Export Wokwi simulation data via MQTT → Node-RED → Azure IoT Hub pipeline

**Step 5: Design Curriculum**
- Module 1: Introduction to DTDL and Digital Twin modeling (weeks 1–2)
- Module 2: IoT device simulation with Wokwi (weeks 3–4)
- Module 3: Building data pipelines with Node-RED (weeks 5–6)
- Module 4: Azure Digital Twins — ingesting live data, querying the twin graph (weeks 7–9)
- Module 5: Analytics and anomaly detection with Azure services (weeks 10–12)
- Module 6: Capstone — design and monitor a smart building or smart factory digital twin (weeks 13–14)

**Step 6: Deliver via OpenEdX**
- Host course materials, assignments, and grading rubrics on [OpenEdX](https://openedx.org/)

#### Feasibility Assessment

| Factor | Rating | Notes |
|---|---|---|
| Cost | ⭐⭐⭐⭐ | Free tier + student credits cover coursework; ~$430–$2,280/year for dept. |
| Implementation Complexity | ⭐⭐⭐⭐ | Azure Portal is GUI-driven and browser-based — accessible to non-technical students. DTDL modeling has a learning curve but is well-documented on Microsoft Learn. Node-RED's visual drag-and-drop interface is highly approachable for beginners. |
| Space Requirements | ⭐⭐⭐⭐⭐ | Fully cloud; zero physical footprint |
| Maintainability | ⭐⭐⭐⭐⭐ | Fully managed by Microsoft; no infrastructure to maintain |
| Curriculum Readiness | ⭐⭐⭐⭐ | Microsoft Learn modules available; reference textbook available |
| Industry Relevance | ⭐⭐⭐⭐⭐ | Azure is widely used in enterprise IoT and smart building sectors |

**Verdict:** One of the two most accessible options for non-technical students. Azure Portal and Node-RED are both GUI/browser-first tools that don't require command-line or coding to get started. DTDL introduces structured modeling concepts gradually. Excellent low-maintenance option with predictable, low annual costs. Best for programs focused on cloud architecture, data engineering, or IoT. Student credits significantly reduce department spend.

#### Getting Started Resources

| Resource | Type | Link |
|---|---|---|
| Azure Digital Twins Overview | Concepts introduction | [Microsoft Learn: What is Azure Digital Twins?](https://learn.microsoft.com/en-us/azure/digital-twins/overview) |
| Quickstart: Azure Digital Twins Explorer | Hands-on beginner quickstart | [ADT Explorer Quickstart](https://learn.microsoft.com/en-us/azure/digital-twins/quickstart-azure-digital-twins-explorer) |
| Quickstart: 3D Scenes Studio | Visualize twins in 3D (robotic arms factory demo) | [3D Scenes Studio Quickstart](https://learn.microsoft.com/en-us/azure/digital-twins/quickstart-3d-scenes-studio) |
| Tutorial: End-to-End Solution | Connect IoT Hub → Digital Twin → Functions | [End-to-End Tutorial](https://learn.microsoft.com/en-us/azure/digital-twins/tutorial-end-to-end) |
| Tutorial: Client App (C#) | Developer-focused coding tutorial | [Client App Tutorial](https://learn.microsoft.com/en-us/azure/digital-twins/tutorial-code) |
| DTDL (Digital Twin Definition Language) | Modeling reference | [DTDL Concepts](https://learn.microsoft.com/en-us/azure/digital-twins/concepts-models) |
| Node-RED Official Documentation | Getting started with flows | [nodered.org/docs](https://nodered.org/docs/) |
| Node-RED Tutorials | Step-by-step beginner series | [nodered.org/docs/tutorials](https://nodered.org/docs/tutorials/) |
| Wokwi + Node-RED + MQTT Tutorial | Simulated IoT end-to-end | [ESP32 + MQTT + Node-RED on Wokwi (Medium)](https://medium.com/@pranavvijayakumar20/building-a-simulated-iot-system-using-esp32-mqtt-node-red-in-wokwi-175e78da28b3) |
| Azure for Students Sign-Up | Free $100 Azure credits | [azure.microsoft.com/free/students](https://azure.microsoft.com/en-us/free/students/) |
| Azure IoT Hub Pricing | Cost reference | [Azure IoT Hub Pricing](https://azure.microsoft.com/en-us/pricing/details/iot-hub/) |
| Azure Digital Twins Pricing | Cost reference | [Azure Digital Twins Pricing](https://azure.microsoft.com/en-us/pricing/details/digital-twins/) |

#### Example End Goal: Smart Building Digital Twin
A student team models a **multi-floor university building** as a digital twin graph in Azure Digital Twins, with rooms, floors, and HVAC zones modeled using DTDL. Wokwi-simulated ESP32 sensors feed temperature and occupancy data via MQTT into a Node-RED pipeline that routes telemetry to Azure IoT Hub. The Azure Digital Twin graph updates in real time and is visualized in both Azure Digital Twins Explorer and a 3D Scenes Studio view of the building floor plan. The final deliverable includes anomaly detection alerts when a room exceeds a set temperature threshold.

**Key skills demonstrated:** DTDL modeling, IoT data pipelines, cloud architecture, Node-RED flow-based programming, event-driven systems, real-time dashboards.

---

### Variation 3: Open-Source IoT Simulation Stack (Low-Cost Entry Point)

#### Overview
This variation builds a lightweight Digital Twins lab using entirely free, open-source tools. It is the **lowest-cost and lowest-complexity** option, and serves as a practical entry point or Phase 1 before scaling to more advanced platforms.

#### Core Tools & Links

| Tool | Role | Cost | Link |
|---|---|---|---|
| Wokwi | Browser-based simulation of Arduino, ESP32, Raspberry Pi Pico | Free | [wokwi.com](https://wokwi.com/) |
| Node-RED | Visual flow-based IoT programming and dashboard | Free, open-source | [nodered.org](https://nodered.org/) |
| Eclipse Ditto | Open-source Digital Twins framework (device state management) | Free, open-source | [eclipse.dev/ditto](https://eclipse.dev/ditto/) |
| InfluxDB | Time-series database for telemetry storage | Free (self-hosted or cloud free tier) | [influxdata.com](https://www.influxdata.com/) |
| Grafana | Real-time telemetry dashboards | Free, open-source | [grafana.com](https://grafana.com/) |
| MQTT (Eclipse Mosquitto) | Lightweight messaging protocol for IoT | Free, open-source | [mosquitto.org](https://mosquitto.org/) |

#### Hardware Requirements & Department Cost Estimate

> ✅ **This is the only variation where STC's existing server room hardware might be partially useful** — but only after a physical inspection confirms minimum viable specs. See Section 4.5 for full assessment.

**Option A: Cloud VM — Recommended Primary (Most Reliable)**

| Item | Quantity | Cost/Month | Annual Cost |
|---|---|---|---|
| Cloud VM (2 vCPU, 4–8 GB RAM, 50 GB SSD) | 1 | ~$10–$20/month | ~$120–$240/year |
| Domain name (optional) | 1 | ~$1/month | ~$12/year |
| **Total Annual (Cloud)** | | | **~$132–$252/year** |

Recommended providers: [DigitalOcean](https://www.digitalocean.com/pricing) (~$12/month for 2 vCPU / 4 GB), [Linode/Akamai](https://www.linode.com/pricing/) (~$12/month), or [AWS Lightsail](https://aws.amazon.com/lightsail/pricing/) (~$10/month).

**Option B: Repurpose Existing Server Room Hardware (Zero Additional Cost — Conditional)**

STC's existing unused server room may host the open-source backend stack at zero cost if the hardware passes a minimum inspection. This eliminates the cloud VM subscription fee entirely.

**Before committing to this path, a faculty member must physically inspect and confirm all of the following:**

| Checklist Item | What to Check | Pass Threshold |
|---|---|---|
| CPU | Model, core count, 64-bit support | Intel Xeon E5 / Core i5+ (2012 era or newer), 4+ cores |
| RAM | Installed RAM amount | ≥ 8 GB (DDR3 acceptable); upgrade to 16 GB recommended (~$20–$40) |
| Storage | HDD vs SSD, free space | ≥ 50 GB free; SSD strongly preferred — add one for ~$50–$80 if missing |
| OS | Can Ubuntu 22.04 LTS be installed? | Must support 64-bit OS; BIOS accessible |
| Network | Connected to campus network? | Must have stable LAN connection |
| Cooling | Is server room ventilated/cooled? | Server room must have working cooling or hardware will overheat |
| Power | Is power supply stable? | UPS (Uninterruptible Power Supply) recommended for classroom reliability |
| Age/condition | Visual inspection | No visible corrosion, capacitor damage, or failed drive indicators |

> **If the hardware fails any of the above checks, fall back immediately to Option A (cloud VM).** Deploying unreliable hardware as classroom infrastructure is a high-risk decision that will disrupt the academic program.

| Item | Cost |
|---|---|
| RAM upgrade (if needed) | ~$20–$40 |
| SSD upgrade (if needed) | ~$50–$80 |
| Ubuntu 22.04 LTS (OS) | Free |
| **Total (Repurpose, conditional)** | **~$0–$120 one-time** |

#### Implementation Steps (Detailed)

**Step 1: Provision the Backend Server or Cloud VM**
- If using cloud: Spin up a VM on DigitalOcean, Linode, or AWS Lightsail (~$10–20/month)
- Install Ubuntu 22.04 LTS
- Install Docker and Docker Compose (simplifies all service deployment)

**Step 2: Deploy Core Services via Docker Compose**
Deploy the following as Docker containers on the server:
- [Eclipse Mosquitto](https://mosquitto.org/) — MQTT broker (port 1883)
- [Eclipse Ditto](https://eclipse.dev/ditto/) — Digital Twin state management (port 8080)
- [InfluxDB](https://www.influxdata.com/) — Time-series telemetry storage (port 8086)
- [Grafana](https://grafana.com/) — Dashboard visualization (port 3000)
- [Node-RED](https://nodered.org/) — Flow-based IoT pipeline (port 1880)

Sample Docker Compose structure is available in the [Eclipse Ditto Docker tutorial](https://eclipse.dev/ditto/installation-operating.html).

**Step 3: Build Sample Lab Exercises with Wokwi**
- Create 3–5 shared Wokwi projects (temperature sensor, servo motor, LED array, DHT22 humidity)
- Configure Wokwi simulators to publish MQTT messages to the classroom MQTT broker
- Students observe real-time state updates flowing through: Wokwi → MQTT → Node-RED → Eclipse Ditto → Grafana dashboard

**Step 4: Design Curriculum**
- Module 1: IoT fundamentals — sensors, MQTT, data protocols (weeks 1–2)
- Module 2: Hardware simulation with Wokwi (weeks 3–4)
- Module 3: Flow-based programming with Node-RED (weeks 5–6)
- Module 4: Digital Twin state management with Eclipse Ditto (weeks 7–9)
- Module 5: Time-series analytics and Grafana dashboards (weeks 10–11)
- Module 6: Capstone — build a full IoT digital twin for a simulated environment (weeks 12–14)

**Step 5: Deliver via OpenEdX**
- Host course materials on [OpenEdX](https://openedx.org/) (can run on the same VM as the lab services)

#### Feasibility Assessment

| Factor | Rating | Notes |
|---|---|---|
| Cost | ⭐⭐⭐⭐⭐ | ~$132–$252/year (cloud) or $300–500 one-time (on-premise) |
| Implementation Complexity | ⭐⭐⭐⭐⭐ | **Most accessible option for non-technical students.** Wokwi runs entirely in a browser — no install. Node-RED uses drag-and-drop visual programming. Grafana dashboards are point-and-click. Docker Compose backend setup is handled by faculty, invisible to students. |
| Space Requirements | ⭐⭐⭐⭐⭐ | No physical footprint (cloud) or a single mini PC |
| Maintainability | ⭐⭐⭐⭐ | All tools are actively maintained open-source projects |
| Curriculum Readiness | ⭐⭐⭐ | Good community resources, but no dedicated teaching kits |
| Industry Relevance | ⭐⭐⭐ | Tools are used in industry but less prestigious than Azure or NVIDIA |

**Verdict:** The **most accessible variation for non-technical students** and the strongest Phase 1 starting point. Wokwi, Node-RED, and Grafana all operate through browser or visual interfaces — no terminal or coding required for students at the introductory level. Faculty handle the Docker backend once; students interact only through dashboards and visual flows. Ideal first course before introducing any coding-heavy tools.

#### Getting Started Resources

| Resource | Type | Link |
|---|---|---|
| Node-RED Getting Started | Official docs: install to first flow | [nodered.org/docs](https://nodered.org/docs/) |
| Node-RED Tutorials | Official beginner tutorials | [nodered.org/docs/tutorials](https://nodered.org/docs/tutorials/) |
| FlowFuse Node-RED Guide (100+ tutorials) | Comprehensive tutorial library | [flowfuse.com/node-red/learn](https://flowfuse.com/node-red/learn/) |
| Node-RED Programming Guide | Lecture-based beginner series | [noderedguide.com](https://noderedguide.com/) |
| Wokwi Getting Started | Simulator intro + example projects | [wokwi.com](https://wokwi.com/) |
| Wokwi + MQTT + ESP32 Tutorial | Full simulated IoT project walkthrough | [Simulating IoT with Wokwi & MQTT (IoTbyHVM)](https://iotbyhvm.ooo/how-to-simulate-an-iot-project-on-wokwi-with-mqtt-and-esp32/) |
| Wokwi + Node-RED + MQTT (end-to-end) | ESP32 sensor → Node-RED dashboard | [Building Simulated IoT with Wokwi & Node-RED (Medium)](https://medium.com/@pranavvijayakumar20/building-a-simulated-iot-system-using-esp32-mqtt-node-red-in-wokwi-175e78da28b3) |
| Eclipse Ditto Getting Started | Official documentation | [eclipse.dev/ditto](https://eclipse.dev/ditto/) |
| Eclipse Ditto Docker Tutorial | Docker Compose deployment guide | [Ditto Installation & Operating](https://eclipse.dev/ditto/installation-operating.html) |
| InfluxDB Getting Started | Time-series database setup | [influxdata.com/get-influxdb](https://www.influxdata.com/get-influxdb/) |
| Grafana Getting Started | Dashboard setup tutorial | [grafana.com/docs/grafana/getting-started](https://grafana.com/docs/grafana/latest/getting-started/) |
| Eclipse Mosquitto MQTT Broker | Broker setup | [mosquitto.org/documentation](https://mosquitto.org/documentation/) |

#### Example End Goal: Smart Factory Floor Monitor
A student team builds a **simulated smart factory monitoring system** using only free, open-source tools. Five Wokwi-simulated ESP32 sensors (temperature, humidity, vibration, proximity) publish MQTT messages to an Eclipse Mosquitto broker hosted on a cloud VM. Node-RED flows route and transform the data, storing it in InfluxDB time-series database and updating Eclipse Ditto digital twin state models. A Grafana dashboard displays live machine health, anomaly alerts, and historical trend graphs. The final deliverable runs entirely in the browser (Wokwi) and on a single cloud VM — no physical hardware required.

**Key skills demonstrated:** MQTT protocol, IoT sensor simulation, flow-based programming, digital twin state management, time-series data storage, real-time dashboarding.

---

### Variation 4: Unity-Based Digital Twin (Visualization-Focused)

#### Overview
This variation uses Unity — the widely adopted game engine — as the primary Digital Twin visualization and simulation platform. It is suitable for students already familiar with Unity and for programs emphasizing real-time 3D and interactive simulation.

#### Core Tools & Links

| Tool | Role | Cost | Link |
|---|---|---|---|
| Unity (Education Plan) | 3D simulation, visualization, Digital Twin runtime | Free for students/educators | [unity.com/education](https://unity.com/education) |
| Unity Digital Twin Toolkit | Connects Unity scenes to live data sources | Included with Unity | [Unity DT Docs](https://docs.unity3d.com/) |
| Node-RED | Data ingestion and pipeline orchestration | Free, open-source | [nodered.org](https://nodered.org/) |
| MQTT (Mosquitto) | Device communication protocol | Free | [mosquitto.org](https://mosquitto.org/) |
| Wokwi | IoT hardware simulation | Free | [wokwi.com](https://wokwi.com/) |

#### Hardware Requirements & Department Cost Estimate

Unity runs on most modern student laptops (Windows or macOS). No specialized GPU is required for basic digital twin scenes.

**Department Infrastructure Cost**

| Item | Quantity | Unit Cost | Annual/One-Time |
|---|---|---|---|
| Unity Education licenses | 0 — free via Unity Student/Educator plan | $0 | $0/year |
| Cloud VM for MQTT + Node-RED backend | 1 VM | ~$10–$20/month | ~$120–$240/year |
| Curriculum development time (custom — no teaching kits exist) | Faculty hours | Internal cost | N/A |
| **Total Annual Operating Cost** | | | **~$120–$240/year** |

> Unity does not provide dedicated Digital Twin teaching kits. Faculty must build curriculum from scratch, which is the primary time cost of this variation.

#### Implementation Steps (Detailed)

**Step 1: Register for Unity Education Plan**
- Visit [unity.com/education](https://unity.com/education) and apply for the Unity Student or Unity Educator license (free)
- Install [Unity Hub](https://unity.com/download) and Unity LTS version on all student machines

**Step 2: Set Up Data Backend**
- Deploy MQTT broker (Mosquitto) and Node-RED on a cloud VM or local server
- Configure Node-RED flows to push IoT telemetry (real or simulated via Wokwi) to a REST API or WebSocket server that Unity can read

**Step 3: Build Digital Twin Scene in Unity**
- Import or create a 3D model of the physical system being twinned (e.g., a factory floor, smart building floor plan, or robotic arm)
- Write Unity C# scripts that poll the data API and update 3D object properties in real time (color, position, rotation, annotations)
- Use Unity's UI system to overlay telemetry values on 3D objects

**Step 4: Design Curriculum**
- Module 1: Unity fundamentals and 3D scene building (weeks 1–3)
- Module 2: IoT data simulation with Wokwi and Node-RED (weeks 4–5)
- Module 3: Connecting live data streams to Unity scenes (weeks 6–8)
- Module 4: Interactive Digital Twin HMI design (weeks 9–11)
- Module 5: Capstone — build a real-time 3D digital twin of a simulated physical system (weeks 12–14)

#### Feasibility Assessment

| Factor | Rating | Notes |
|---|---|---|
| Cost | ⭐⭐⭐⭐⭐ | Near-zero; free Unity education license |
| Implementation Complexity | ⭐⭐ | Unity's GUI is visual but requires C# scripting to connect live data — a significant barrier for non-technical students with no coding background. Unsuitable as an introductory course; better positioned as an intermediate track for students with prior programming exposure. |
| Space Requirements | ⭐⭐⭐⭐⭐ | No physical footprint |
| Maintainability | ⭐⭐⭐⭐ | Unity is well-supported; large developer community |
| Curriculum Readiness | ⭐⭐⭐ | No dedicated Digital Twin teaching kits; custom curriculum required |
| Industry Relevance | ⭐⭐⭐⭐ | Used in manufacturing, AEC, and defense digital twin deployments |

**Verdict:** The 3D visualization payoff is compelling, but C# scripting for live data integration makes this **unsuitable as a first course for non-technical students**. Recommended as an intermediate elective for students who have completed a foundational coding course. If STC's student population grows in technical depth over time, this becomes a stronger option. Building curriculum from scratch remains the main faculty overhead.

#### Getting Started Resources

| Resource | Type | Link |
|---|---|---|
| Unity Education Plan (Free License) | License registration | [unity.com/education](https://unity.com/education) |
| Unity Learn: Getting Started | Official beginner learning platform | [learn.unity.com](https://learn.unity.com/) |
| Unity Learn: Junior Programmer Pathway | Structured beginner course | [Unity Junior Programmer Pathway](https://learn.unity.com/pathway/junior-programmer) |
| Unity Manual: Getting Started | Official reference documentation | [docs.unity3d.com/Manual](https://docs.unity3d.com/Manual/index.html) |
| Unity Download (Unity Hub) | Install guide | [unity.com/download](https://unity.com/download) |
| Node-RED + MQTT backend setup | Data pipeline for Unity | [nodered.org/docs](https://nodered.org/docs/) |
| Unity Education: Licensing FAQ | Verify free access eligibility | [unity.com/education/faq](https://unity.com/education/faq) |

#### Example End Goal: Interactive 3D Server Room Digital Twin
A student team builds an **interactive 3D digital twin of a university server room** in Unity. Physical server rack models (created in Unity or imported as free assets) are linked via C# scripts to a live telemetry API fed by Node-RED and a Wokwi-simulated temperature/humidity sensor array. When a simulated server rack exceeds a temperature threshold, its 3D model changes color in real time from green → yellow → red, and a dashboard overlay displays the alert. The final deliverable is a shareable Unity build that instructors can run on any Windows or macOS machine.

**Key skills demonstrated:** Unity 3D scene assembly, real-time data binding, C# scripting, IoT integration, interactive HMI design, visual simulation.

---

### Variation 5: XMPro – Industrial No-Code Platform

#### Overview
[XMPro](https://xmpro.com/) is an enterprise-grade, no-code/low-code Digital Twins platform used in real industrial deployments (energy, mining, manufacturing). It is oriented toward **Track B (Industrial Analytics)** and targets operational decision-making at scale.

#### Core Tools & Links

| Tool | Role | Cost | Link |
|---|---|---|---|
| XMPro iBOS | End-to-end Digital Twin platform (data streams, AI agents, dashboards) | Enterprise annual subscription — pricing not public | [xmpro.com/pricing](https://xmpro.com/pricing/) |
| XMPro + Azure Digital Twins | XMPro can integrate with Azure DT as the underlying twin graph | Requires both licenses | [XMPro + Azure DT](https://xmpro.com/microsoft-azure-digital-twins-everything-you-need-to-know/) |
| XMPro Free Trial | Available for business evaluation only — **not for students** | Free (business only) | [xmpro.com/free-trial](https://xmpro.com/free-trial/) |

> **Critical note on academic access:** XMPro's free trial is explicitly restricted to business use. Per their website: *"Students interested in learning about XMPro should request for their university to obtain a license for educational purposes."* This means STC must negotiate an institutional license directly with XMPro. No public academic pricing is listed. This variation depends entirely on a successful licensing agreement.

#### Hardware Requirements & Department Cost Estimate

XMPro is cloud-hosted SaaS — no hardware required.

| Item | Cost |
|---|---|
| XMPro institutional license | Unknown — requires direct negotiation with XMPro sales |
| Azure infrastructure (if using XMPro + Azure DT integration) | ~$300–$2,000/year depending on usage |
| Student laptops | $0 (existing) |
| **Estimated Annual Cost** | **Unknown — high risk without confirmed pricing** |

#### Implementation Steps (Conditional on Licensing)

**Step 1: Contact XMPro for Academic Licensing**
- Email XMPro sales at [xmpro.com/contact](https://xmpro.com/contact-us/)
- Request an academic/educational institutional license for STC
- Inquire about sandbox environments for student use

**Step 2 (if licensed): Provision XMPro Environment**
- XMPro provisions a cloud tenant for STC
- Faculty complete XMPro's onboarding training (typically 1–2 days for basic use)
- Connect XMPro data streams to simulated data sources (Node-RED or Wokwi)

**Step 3: Design Curriculum Around XMPro Capabilities**
- Module 1: Industrial Digital Twin concepts and XMPro architecture (weeks 1–2)
- Module 2: Building data streams with XMPro's drag-and-drop interface (weeks 3–5)
- Module 3: AI agents and recommendation systems in XMPro (weeks 6–8)
- Module 4: Building operational dashboards and real-time KPI views (weeks 9–11)
- Module 5: Capstone — design a condition monitoring or predictive maintenance digital twin (weeks 12–14)

#### Feasibility Assessment

| Factor | Rating | Notes |
|---|---|---|
| Cost | ⭐⭐ | Enterprise pricing; academic licensing unconfirmed — high budget risk |
| Implementation Complexity | ⭐⭐⭐⭐⭐ | **Most accessible for non-technical students among all tools** — XMPro is a no-code/low-code drag-and-drop platform requiring no programming. However, this rating is conditional on securing academic licensing, which remains unconfirmed. |
| Space Requirements | ⭐⭐⭐⭐⭐ | Cloud SaaS; no physical footprint |
| Maintainability | ⭐⭐⭐⭐⭐ | Fully managed SaaS |
| Curriculum Readiness | ⭐⭐ | Limited public academic resources; no teaching kits |
| Industry Relevance | ⭐⭐⭐⭐⭐ | Highly relevant in heavy industry sectors |

**Verdict:** High industry relevance and low technical complexity, but cost and curriculum gaps make this a high-risk option without a confirmed academic licensing agreement. **Do not commit to this variation without first securing a license.** Recommend reaching out to XMPro before including it in the implementation plan.

#### Getting Started Resources

| Resource | Type | Link |
|---|---|---|
| XMPro Product Overview | Platform introduction | [xmpro.com](https://xmpro.com/) |
| XMPro Pricing Page | Licensing model details | [xmpro.com/pricing](https://xmpro.com/pricing/) |
| XMPro Free Trial (Business Only) | Trial terms — academic exclusion noted | [xmpro.com/free-trial](https://xmpro.com/free-trial/) |
| XMPro + Azure Digital Twins Guide | Integration walkthrough | [XMPro + Azure DT](https://xmpro.com/microsoft-azure-digital-twins-everything-you-need-to-know/) |
| XMPro Ultimate Digital Twin Guide | Concept deep dive | [xmpro.com/ultimate-guide](https://xmpro.com/digital-twins-the-ultimate-guide/) |
| XMPro Contact / Academic Inquiry | Request academic licensing | [xmpro.com/contact-us](https://xmpro.com/contact-us/) |
| XMPro YouTube Channel | Product demos and walkthroughs | [XMPro YouTube](https://www.youtube.com/@xmpro) |

> ⚠️ **Feasibility Flag:** No public tutorial or academic getting-started guide exists for XMPro. All onboarding requires a vendor-provisioned environment. This is a blocking dependency — STC cannot self-provision a trial. Contact XMPro before this variation can be properly evaluated.

#### Example End Goal: Predictive Maintenance Dashboard for Industrial Pumps
A student team uses XMPro's drag-and-drop interface to build a **predictive maintenance digital twin** for a simulated industrial pump system. Live telemetry streams (vibration, temperature, flow rate) from a simulated data source are ingested via XMPro data streams. An XMPro AI agent monitors the streams, detects early signs of bearing failure using a trained ML model, and generates prescriptive maintenance recommendations. The final deliverable is an XMPro operational dashboard showing real-time pump health KPIs and automated maintenance work orders.

**Key skills demonstrated:** Industrial IoT data streams, no-code AI agent design, operational dashboarding, prescriptive analytics, enterprise digital twin architecture.

---

### Variation 6: AMD Ecosystem (Open-Source Simulation Alternative)

#### Overview
AMD offers a compelling open-source alternative to NVIDIA's simulation stack, built around the **ROCm** GPU computing platform, the **Genesis** robotics simulator, and the **AMD University Program** — which provides free teaching materials, discounted hardware, and AI training specifically for academic institutions. This variation is oriented toward **Track A (Robotics & Simulation)** and is best understood as a cost-conscious, open-source counterpart to Variation 1.

A key differentiator: AMD published a dedicated blog post in February 2026 — *[Digital Twins on AMD: Building Robotic Simulations Using Edge AI PCs](https://rocm.blogs.amd.com/artificial-intelligence/rocm-genesis/README.html)* — demonstrating that high-fidelity robotic simulation and digital twins can run on a **single AMD Ryzen AI MAX laptop**, without needing a separate GPU workstation. This has direct relevance to STC's space and hardware constraints.

#### Core Tools & Links

| Tool | Role | Cost | Link |
|---|---|---|---|
| AMD ROCm | Open-source GPU computing platform (AMD's equivalent to NVIDIA CUDA) | Free, open-source | [ROCm Docs](https://rocm.docs.amd.com/) |
| Genesis | Open-source physics-based robotics simulation platform | Free, open-source | [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis) |
| AMD University Program (AUP) | Free teaching materials, discounted hardware, AI training for academics | Free (registration required) | [amd.com/university](https://www.amd.com/en/corporate/university-program.html) |
| AUP Learning Cloud | Pre-configured Jupyter-based AI courses incl. "Physical Simulation" | Free, open-source | [AUP GitHub](https://github.com/AMDResearch/aup-learning-cloud) |
| AMD AI Developer Program | Developer tools, community resources, contest access | Free | [AMD AI Dev Program](https://www.amd.com/en/developer/ai-dev-program.html) |
| ROCm Blogs – Robotics | Tutorial library including Digital Twins and robotic simulation | Free | [ROCm Robotics Blogs](https://rocm.blogs.amd.com/robotics.html) |
| PyTorch (ROCm build) | AI/ML framework running natively on AMD GPUs | Free, open-source | [PyTorch ROCm](https://pytorch.org/get-started/locally/) |
| Robotec.ai RoSi (via AMD partnership) | Open-source digital twin simulation for autonomous systems | Free, open-source | [Robotec.ai](https://robotec.ai/) |
| ROS 2 | Robotics middleware (same as NVIDIA variation — hardware-agnostic) | Free, open-source | [docs.ros.org](https://docs.ros.org/en/humble/index.html) |
| Wokwi | Browser-based IoT hardware simulation | Free | [wokwi.com](https://wokwi.com/) |

> **AMD University Program note:** The AUP provides educators and researchers with ready-to-use teaching materials, hands-on tutorials, and flexible open-source compute environments that can run locally on AMD-powered systems or in AMD's remote HPC cluster. Faculty can apply at [amd.com/university-program](https://www.amd.com/en/corporate/university-program.html).

> **Student contest opportunity:** The [AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023) offers a **$2,500 USD prize specifically for university students** in the AMD University Program category (Generative AI, Robotics AI, PC AI tracks). This is a direct funding opportunity for STC students and could subsidize lab hardware costs.

#### What Students Learn
- Robotic simulation using the Genesis physics engine on AMD hardware
- GPU-accelerated AI and machine learning using ROCm (open-source CUDA alternative)
- Sim-to-real transfer concepts using Genesis + ROS 2
- Parallel simulation environments for robot policy training
- Digital twin concepts applied to industrial and robotics contexts
- Open-source AI stack (PyTorch on ROCm) — hardware-vendor-agnostic skills

#### How AMD Compares to NVIDIA for This Use Case

| Capability | NVIDIA (V1) | AMD (V6) |
|---|---|---|
| Simulation platform | Isaac Sim (NVIDIA-proprietary) | Genesis (open-source, hardware-agnostic) |
| GPU computing | CUDA (proprietary) | ROCm (open-source) |
| Teaching kits | NVIDIA DLI kits (polished, extensive) | AUP Learning Cloud (growing, open-source) |
| Hardware requirement | Dedicated NVIDIA GPU (RTX) | AMD Ryzen AI MAX laptop or discrete Radeon GPU |
| Ecosystem maturity | Very high (industry standard) | Growing rapidly; still behind NVIDIA |
| Vendor lock-in | High (CUDA/NVIDIA-only) | Low (open-source, portable) |
| 3D visualization depth | Very high (Omniverse) | Moderate (Genesis is lighter) |
| Cost | Free software; higher hardware cost | Free software; lower hardware cost |

#### Hardware Requirements & Department Cost Estimate

> ⚠️ **Space constraint reminder:** STC has no student lab space. On-premise workstation and Radeon GPU workstation options below are future investment scenarios. Cloud and AUP HPC options are the only immediately feasible paths.

**Option A: AMD AUP HPC Cluster Access (Zero Hardware — Recommended First)**
The AMD University Program may provide remote HPC cluster access to qualified institutions at no cost. Students run Genesis simulations through a browser-based Jupyter interface — no local GPU needed.

| Item | Cost |
|---|---|
| AUP registration & HPC cluster access | Free (application required) |
| Student laptops (existing) | $0 |
| **Total** | **$0** |

> Apply at [amd.com/university-program](https://www.amd.com/en/corporate/university-program.html). HPC access is not guaranteed — confirm availability before building curriculum around it.

**Option B: AMD Ryzen AI MAX Laptops (Future Investment — No Lab Space Needed)**
These are standard laptops that double as simulation machines. No dedicated lab space required — students use them as their primary device.

| Item | Quantity | Unit Cost | Total |
|---|---|---|---|
| AMD Ryzen AI MAX 395 laptop (e.g., ASUS ProArt, Lenovo ThinkPad X1 Extreme) | 3–5 units (faculty + demo) | ~$1,500–$2,000 | ~$4,500–$10,000 |
| **Total (Laptops)** | | | **~$4,500–$10,000** |

**Option C: AMD Radeon Workstations (Future Investment — Requires Lab Space STC Does Not Currently Have)**

| Item | Quantity | Unit Cost | Total |
|---|---|---|---|
| AMD Radeon RX 7900 XTX workstation | 2–3 units | ~$1,800–$2,500 | ~$3,600–$7,500 |
| Monitor | 2–3 units | ~$300–$400 | ~$600–$1,200 |
| **Total (Workstations, future)** | | | **~$4,200–$8,700** |

> **Old server room hardware note:** STC's existing server room has **no graphics card**. ROCm requires an AMD GPU to function — the old hardware cannot run Genesis or any ROCm-accelerated workload. Cloud (AUP HPC) or new AMD hardware are the only viable paths for this variation.

#### Implementation Steps (Detailed)

**Step 1: Register for the AMD University Program**
- Apply at [amd.com/en/corporate/university-program.html](https://www.amd.com/en/corporate/university-program.html)
- Request access to: teaching materials, AUP Learning Cloud, and AMD HPC cluster
- Register faculty for the [AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html)

**Step 2: Set Up the AUP Learning Cloud**
- Clone the [AUP Learning Cloud repository](https://github.com/AMDResearch/aup-learning-cloud)
- Run the single-node installer (Ubuntu 24.04):
  ```bash
  git clone https://github.com/AMDResearch/aup-learning-cloud.git
  cd aup-learning-cloud/deploy/
  sudo ./single-node.sh install
  ```
- Access pre-built courses at `http://localhost:30890` — including the **"Physical Simulation"** course that directly covers Genesis + ROCm digital twin workflows

**Step 3: Install ROCm and Genesis**
- Install [ROCm 7.x](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) on Ubuntu 24.04 (AMD hardware required)
- Install [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) robotics simulator:
  ```bash
  pip install genesis-world
  ```
- Test with Genesis's built-in robot environments (Franka arm, quadruped, locomotion tasks)

**Step 4: Connect Genesis to ROS 2**
- Install ROS 2 Humble (hardware-agnostic, same as NVIDIA variation)
- Configure Genesis as the simulation backend for ROS 2 robot nodes
- Build sample lab: Genesis simulated robot → ROS 2 control → Grafana telemetry dashboard

**Step 5: Design Curriculum Around AUP Materials**
- Module 1: Introduction to ROCm and GPU-accelerated AI (weeks 1–2) — use AUP Learning Cloud courses
- Module 2: Physics-based simulation fundamentals with Genesis (weeks 3–4)
- Module 3: Building a robot digital twin scene in Genesis (weeks 5–7)
- Module 4: Parallel simulation and robot policy training (weeks 8–9)
- Module 5: ROS 2 integration — sim-to-real transfer concepts (weeks 10–11)
- Module 6: Capstone — build and demonstrate a complete robotic digital twin (weeks 12–14)

**Step 6: Submit to AMD Pervasive AI Contest (Optional but Recommended)**
- Encourage student teams to enter the [AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023) in the Robotics AI category
- AMD University Program prize: $2,500 — potential partial hardware funding

#### Feasibility Assessment

| Factor | Rating | Notes |
|---|---|---|
| Cost | ⭐⭐⭐⭐⭐ | Software fully free; hardware cheaper than NVIDIA; AUP HPC option costs $0 |
| Implementation Complexity | ⭐⭐ | **High barrier for non-technical students.** ROCm requires Linux installation, terminal commands, and Python environment setup. Genesis is Python-based. The AUP Learning Cloud's Jupyter-based courses lower the barrier somewhat, but this variation still assumes basic programming literacy. Suitable as an advanced elective, not an introductory course. |
| Space Requirements | ⭐⭐⭐⭐⭐ | Laptops or AUP cloud — no dedicated lab space needed |
| Maintainability | ⭐⭐⭐⭐ | Open-source stack; AMD actively maintains ROCm; Genesis is community-maintained |
| Curriculum Readiness | ⭐⭐⭐⭐ | AUP Learning Cloud provides structured courses; less polished than NVIDIA DLI |
| Industry Relevance | ⭐⭐⭐⭐ | Growing rapidly; ROCm + open-source stack increasingly valued in industry |

**Verdict:** Strong budget-conscious alternative to NVIDIA with genuine open-source advantages, but **not appropriate as an introductory course for non-technical students**. ROCm and Genesis both require Python and Linux comfort. Recommended as a **parallel advanced track alongside V1**, or as the simulation option when hardware budget is tight and students already have programming foundations. The AMD University Program HPC cluster access remains the lowest-cost hardware option of all variations.

#### Getting Started Resources

| Resource | Type | Link |
|---|---|---|
| AMD University Program (Register) | Academic program enrollment | [amd.com/university-program](https://www.amd.com/en/corporate/university-program.html) |
| AMD AI Developer Program | Developer tools & community | [amd.com/ai-dev-program](https://www.amd.com/en/developer/ai-dev-program.html) |
| AUP Learning Cloud (GitHub) | Pre-configured AI courses incl. Physical Simulation | [AUP Learning Cloud](https://github.com/AMDResearch/aup-learning-cloud) |
| Genesis Documentation | Official docs: install, tutorials, API | [genesis-world.readthedocs.io](https://genesis-world.readthedocs.io/) |
| Genesis User Guide | Getting Started section | [Genesis User Guide](https://genesis-world.readthedocs.io/en/latest/user_guide/) |
| Genesis: Control Your Robot | Hands-on robot control tutorial | [Control Your Robot (Genesis)](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html) |
| Genesis Project Homepage | Overview, demos, capabilities | [genesis-embodied-ai.github.io](https://genesis-embodied-ai.github.io/) |
| Genesis GitHub Repository | Source code and examples | [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis) |
| DataCamp: Genesis Setup Tutorial | Step-by-step install guide (Google Colab compatible) | [DataCamp Genesis Tutorial](https://www.datacamp.com/tutorial/genesis-physics-engine-tutorial) |
| DataCamp: Genesis Guide with Examples | Overview and use cases | [DataCamp Genesis Guide](https://www.datacamp.com/blog/genesis-physics-engine) |
| AMD ROCm Digital Twins Blog | AMD tutorial: Genesis + Ryzen AI MAX | [Digital Twins on AMD (ROCm Blog)](https://rocm.blogs.amd.com/artificial-intelligence/rocm-genesis/README.html) |
| AMD ROCm Documentation | ROCm install and developer reference | [rocm.docs.amd.com](https://rocm.docs.amd.com/) |
| AMD ROCm Robotics Blog Index | Robotics tutorials on AMD hardware | [ROCm Robotics Blogs](https://rocm.blogs.amd.com/robotics.html) |
| PyTorch + ROCm Installation | Framework setup on AMD GPUs | [PyTorch: Get Started Locally](https://pytorch.org/get-started/locally/) |
| AMD Pervasive AI Developer Contest | Student contest ($2,500 university prize) | [AMD AI Contest (Hackster.io)](https://www.hackster.io/contests/amd2023) |
| AMD AI DevDay 2026 | AUP curriculum and deployment options | [AMD AI DevDay 2026](https://www.amd.com/en/corporate/events/amd-ai-dev-day.html) |

#### Example End Goal: Quadruped Robot Navigation Digital Twin on AMD Hardware
A student team uses Genesis and the AUP Learning Cloud's Physical Simulation course to build a **digital twin of a quadruped robot navigating varied terrain**. Running entirely on AMD Ryzen AI MAX hardware (or the AUP HPC cluster), the team trains a locomotion policy using PyTorch on ROCm via parallel simulation environments — hundreds of simulated robots training simultaneously. The trained policy is tested on different terrains (flat, rough, inclined), and the simulation results are visualized and recorded. The final deliverable is a reproducible Jupyter notebook and Genesis scene demonstrating sim-to-real locomotion concepts — runnable on any AMD GPU or Google Colab.

**Key skills demonstrated:** Open-source GPU computing (ROCm), physics-based simulation (Genesis), reinforcement learning for robotics, parallel environments, sim-to-real concepts, PyTorch on non-NVIDIA hardware.

---

## 7. Comparative Analysis

### 7.1 Department Cost Summary

| Variation | Year 1 Cost (Dept.) | Ongoing Annual Cost | Hardware Required | Feasible Now? |
|---|---|---|---|---|
| **V1: NVIDIA (Omniverse Cloud)** | ~$0–TBD | ~$0–TBD | None — browser-based | ✅ Yes — pending NVIDIA academic cloud pricing |
| **V1: NVIDIA (On-Premise, 3 stations)** | ~$8,900–$25,500 | ~$0 | 3–5 NVIDIA GPU workstations | ❌ No lab space; no GPU; future only |
| **V1: NVIDIA (Hybrid, 2 stations)** | ~$6,400 | ~$0 | 2 NVIDIA GPU workstations | ❌ No lab space; future only |
| **V2: Azure Digital Twins** | ~$430–$2,280 | ~$430–$2,280/year | None — cloud + student laptops | ✅ Yes — immediately feasible |
| **V3: Open-Source (Cloud VM)** | ~$132–$252 | ~$132–$252/year | None — cloud + student laptops | ✅ Yes — immediately feasible |
| **V3: Open-Source (Old Server Room)** | ~$0–$120 one-time | ~$0 | Old server room (conditional on inspection) | ⚠️ Conditional — inspect hardware first; no GPU needed for V3 |
| **V4: Unity** | ~$120–$240 | ~$120–$240/year | None for basic; GPU needed for complex scenes | ⚠️ Partial — basic scenes OK on student laptops; no GPU means limited simulation fidelity |
| **V5: XMPro** | Unknown | Unknown | None — cloud SaaS | ⚠️ Conditional — requires confirmed academic license |
| **V6: AMD (AUP HPC Cluster)** | ~$0 | ~$0 | None — remote HPC | ✅ Possible — pending AUP application approval |
| **V6: AMD (Ryzen AI MAX Laptops)** | ~$4,500–$10,000 | ~$0 | 3–5 AMD Ryzen AI MAX laptops | ⚠️ No lab space needed; requires budget approval |
| **V6: AMD (Radeon Workstations)** | ~$4,200–$8,700 | ~$0 | 2–3 AMD Radeon GPU workstations | ❌ No lab space; no GPU; future only |

### 7.2 Variation Scorecard

| Factor (Weight) | V1: NVIDIA | V2: Azure | V3: Open-Source | V4: Unity | V5: XMPro | V6: AMD |
|---|---|---|---|---|---|---|
| **Cost** (20%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Student Accessibility** (25%) | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐* | ⭐⭐ |
| **Low Faculty Complexity** (15%) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Space Requirements** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Maintainability** (10%) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Curriculum Readiness** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Industry Relevance** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

> *V5 XMPro accessibility rating is conditional on academic licensing being confirmed. Without a licensed environment, XMPro cannot be evaluated or used by students.

> **Note on weighting change:** Student Accessibility has been elevated to the highest single weight (25%) given that most STC students have non-technical or limited programming backgrounds. This is the primary feasibility constraint for this institution.

### 7.4 Example End Goals by Path

| Variation | Capstone End Goal | Skills Showcased |
|---|---|---|
| **V1: NVIDIA** | Autonomous warehouse robot arm: trained in Isaac Lab, visualized in Omniverse, deployed via ROS 2 | RL training, sim-to-real, OpenUSD, synthetic data |
| **V2: Azure** | Smart building digital twin: live sensor telemetry → Azure IoT Hub → DTDL twin graph → 3D dashboard | DTDL modeling, cloud IoT, Node-RED pipelines |
| **V3: Open-Source** | Smart factory floor monitor: 5 simulated sensors → MQTT → Node-RED → Eclipse Ditto → Grafana | MQTT, IoT data flows, open-source DT management |
| **V4: Unity** | Interactive 3D server room twin: live telemetry data drives real-time color and alert overlays in Unity | Unity 3D, C# data binding, HMI visualization |
| **V5: XMPro** | Predictive maintenance dashboard: AI agent monitors pump telemetry → auto-generates work orders | No-code AI streams, prescriptive analytics |
| **V6: AMD** | Quadruped robot navigation: parallel RL training on AMD GPU → terrain navigation policy via Genesis | ROCm, Genesis, PyTorch, open-source sim-to-real |

### 7.5 Link Verification Index

All links in this document are live as of May 2026. The table below consolidates every external link for manual verification by reviewers.

| Category | Resource | URL |
|---|---|---|
| **NVIDIA** | Omniverse Platform | https://www.nvidia.com/en-us/omniverse/ |
| **NVIDIA** | Isaac Sim Docs | https://docs.omniverse.nvidia.com/isaacsim/latest/index.html |
| **NVIDIA** | Isaac Sim Quickstart | https://docs.isaacsim.omniverse.nvidia.com/4.5.0/introduction/quickstart_index.html |
| **NVIDIA** | Getting Started with Isaac Sim (Full Path) | https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/ |
| **NVIDIA** | Building Your First Robot | https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/building-your-first-robot-in-isaac-sim/index.html |
| **NVIDIA** | Isaac Lab GitHub | https://github.com/isaac-sim/IsaacLab |
| **NVIDIA** | Isaac ROS Getting Started | https://nvidia-isaac-ros.github.io/getting_started/index.html |
| **NVIDIA** | OpenUSD Docs | https://docs.nvidia.com/learn-openusd/latest/index.html |
| **NVIDIA** | Sim-to-Real SO-101 Course | https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/01-overview.html |
| **NVIDIA** | Synthetic Data Generation | https://www.nvidia.com/en-us/use-cases/synthetic-data-generation-for-agentic-ai/ |
| **NVIDIA** | DLI Teaching Kits | https://www.nvidia.com/en-us/training/educator-programs/teaching-kits/ |
| **NVIDIA** | Edge AI & Robotics Kit (Bitbucket) | https://bitbucket.org/nvidia-dli/edge-ai-and-robotics-teaching-kit-labs/src/master/ |
| **NVIDIA** | Generative AI Teaching Kit | https://bitbucket.org/nvidia-dli/generative-ai-teaching-kit-solutions/src/main/ |
| **NVIDIA** | Education Pricing Program | https://developer.nvidia.com/education-pricing |
| **NVIDIA** | Omniverse License Agreement | https://docs.omniverse.nvidia.com/enterprise/latest/common/NVIDIA_Omniverse_License_Agreement.html |
| **Azure** | Azure Digital Twins Overview | https://learn.microsoft.com/en-us/azure/digital-twins/overview |
| **Azure** | ADT Explorer Quickstart | https://learn.microsoft.com/en-us/azure/digital-twins/quickstart-azure-digital-twins-explorer |
| **Azure** | 3D Scenes Studio Quickstart | https://learn.microsoft.com/en-us/azure/digital-twins/quickstart-3d-scenes-studio |
| **Azure** | End-to-End Tutorial | https://learn.microsoft.com/en-us/azure/digital-twins/tutorial-end-to-end |
| **Azure** | Client App Tutorial (C#) | https://learn.microsoft.com/en-us/azure/digital-twins/tutorial-code |
| **Azure** | DTDL Concepts | https://learn.microsoft.com/en-us/azure/digital-twins/concepts-models |
| **Azure** | ADT Pricing | https://azure.microsoft.com/en-us/pricing/details/digital-twins/ |
| **Azure** | IoT Hub Pricing | https://azure.microsoft.com/en-us/pricing/details/iot-hub/ |
| **Azure** | Azure for Students | https://azure.microsoft.com/en-us/free/students/ |
| **Node-RED** | Official Docs | https://nodered.org/docs/ |
| **Node-RED** | Official Tutorials | https://nodered.org/docs/tutorials/ |
| **Node-RED** | FlowFuse Guide (100+ tutorials) | https://flowfuse.com/node-red/learn/ |
| **Node-RED** | Programming Guide | https://noderedguide.com/ |
| **Wokwi** | Simulator Homepage | https://wokwi.com/ |
| **Wokwi** | MQTT + ESP32 Tutorial | https://iotbyhvm.ooo/how-to-simulate-an-iot-project-on-wokwi-with-mqtt-and-esp32/ |
| **Wokwi** | Wokwi + Node-RED + MQTT (Medium) | https://medium.com/@pranavvijayakumar20/building-a-simulated-iot-system-using-esp32-mqtt-node-red-in-wokwi-175e78da28b3 |
| **Eclipse** | Eclipse Ditto | https://eclipse.dev/ditto/ |
| **Eclipse** | Ditto Docker Deployment | https://eclipse.dev/ditto/installation-operating.html |
| **Eclipse** | Mosquitto MQTT Broker | https://mosquitto.org/ |
| **InfluxDB** | Getting Started | https://www.influxdata.com/get-influxdb/ |
| **Grafana** | Getting Started | https://grafana.com/docs/grafana/latest/getting-started/ |
| **Unity** | Education Plan | https://unity.com/education |
| **Unity** | Unity Learn Platform | https://learn.unity.com/ |
| **Unity** | Unity Download (Hub) | https://unity.com/download |
| **Unity** | Unity Manual | https://docs.unity3d.com/Manual/index.html |
| **XMPro** | Platform Homepage | https://xmpro.com/ |
| **XMPro** | Pricing | https://xmpro.com/pricing/ |
| **XMPro** | Free Trial (Academic Exclusion) | https://xmpro.com/free-trial/ |
| **XMPro** | Contact / Academic Inquiry | https://xmpro.com/contact-us/ |
| **XMPro** | + Azure DT Integration | https://xmpro.com/microsoft-azure-digital-twins-everything-you-need-to-know/ |
| **XMPro** | Ultimate Digital Twin Guide | https://xmpro.com/digital-twins-the-ultimate-guide/ |
| **AMD** | University Program | https://www.amd.com/en/corporate/university-program.html |
| **AMD** | AI Developer Program | https://www.amd.com/en/developer/ai-dev-program.html |
| **AMD** | Digital Twins on AMD (ROCm Blog) | https://rocm.blogs.amd.com/artificial-intelligence/rocm-genesis/README.html |
| **AMD** | ROCm Robotics Blog Index | https://rocm.blogs.amd.com/robotics.html |
| **AMD** | ROCm Documentation | https://rocm.docs.amd.com/ |
| **AMD** | AUP Learning Cloud (GitHub) | https://github.com/AMDResearch/aup-learning-cloud |
| **AMD** | Pervasive AI Contest (Hackster) | https://www.hackster.io/contests/amd2023 |
| **AMD** | AMD AI DevDay 2026 | https://www.amd.com/en/corporate/events/amd-ai-dev-day.html |
| **Genesis** | Homepage | https://genesis-embodied-ai.github.io/ |
| **Genesis** | Documentation | https://genesis-world.readthedocs.io/ |
| **Genesis** | User Guide | https://genesis-world.readthedocs.io/en/latest/user_guide/ |
| **Genesis** | Robot Control Tutorial | https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html |
| **Genesis** | GitHub Repository | https://github.com/Genesis-Embodied-AI/Genesis |
| **Genesis** | DataCamp Setup Tutorial | https://www.datacamp.com/tutorial/genesis-physics-engine-tutorial |
| **PyTorch** | ROCm Installation Guide | https://pytorch.org/get-started/locally/ |
| **OpenEdX** | Platform Homepage | https://openedx.org/ |
| **OpenEdX** | Tutor Deployment | https://docs.tutor.edly.io/ |
| **ROS 2** | Official Documentation | https://docs.ros.org/en/humble/index.html |
| **Books** | Building Industrial Digital Twins (Nath, 2021) | https://www.packtpub.com/product/building-industrial-digital-twins/9781839219535 |

### 7.6 Best Fit by Scenario

| Scenario | Best Variation |
|---|---|
| Non-technical students, zero budget, start now | **V3: Open-Source Stack** (most accessible, no coding required) |
| Non-technical students, cloud-first | **V2: Azure Digital Twins** (GUI/browser-driven, guided tutorials) |
| Mixed ability cohort (non-tech majority + some tech) | **V3 (all students) + V2 (intermediate) + V1 or V6 (advanced elective)** |
| Zero budget, AMD HPC access approved | V6: AMD (advanced students only) |
| Dean-approved budget, maximum industry impact | V1: NVIDIA Hybrid (advanced track) |
| Budget-conscious simulation track (advanced students) | V6: AMD Ryzen AI MAX |
| No-code industrial platform, licensing confirmed | V5: XMPro (most accessible tool, highest license risk) |
| Phased approach, low risk | **V3 → V2 → V1 or V6** |
| NVIDIA vs AMD comparison lab (senior elective) | V1 + V6 side-by-side |

---

## 8. Recommended Approach

Based on STC's confirmed infrastructure constraints — **no dedicated lab space, no GPU hardware, a decade-old server room, and a student majority with non-technical backgrounds** — the recommended approach prioritizes what is immediately feasible with zero infrastructure investment, then builds toward more advanced capabilities only as resources allow.

**Immediately feasible (today, ~$10–$20/month):** V3 Open-Source Stack, V2 Azure Digital Twins
**Feasible with cloud access confirmed:** V1 NVIDIA (Omniverse Cloud), V6 AMD (AUP HPC)
**Feasible with future budget + space:** V1/V6 on-premise hardware options
**Conditional:** V5 XMPro (licensing), V3 old server room (inspection required)

The guiding principle is *accessibility first, depth second*: start every student on visual, browser-based tools, then progressively introduce technical depth for students who are ready.

### Phase 1 (Immediate, ~$0–$252/year): Open-Source Foundation — All Students
Deploy the open-source stack (Wokwi + Node-RED + Eclipse Ditto + Grafana) on a low-cost cloud VM. Students interact exclusively through browser-based and visual interfaces. No terminal or coding required at the introductory level.

This phase is suitable for **all students regardless of technical background** and serves as the universal on-ramp to Digital Twins concepts. It answers the question: *"What is a digital twin and how does data flow through it?"*

### Phase 2A (Semester 2, ~$430–$2,280/year): Azure Digital Twins — Intermediate Track
Introduce Azure Digital Twins for students who have completed Phase 1. Azure Portal and DTDL modeling are GUI-driven and supported by structured Microsoft Learn tutorials. Node-RED, already introduced in Phase 1, connects directly to Azure IoT Hub, minimizing the learning leap.

This phase adds: cloud architecture, structured data modeling, and event-driven systems — accessible to non-technical students with Phase 1 foundations.

### Phase 2B (Semester 2–3, ~$0 software): NVIDIA / AMD Simulation — Advanced Track (Elective)
Introduce NVIDIA Omniverse or AMD Genesis for students with prior coding experience or those who excelled in Phase 1 and Phase 2A. This is positioned as an **advanced elective or capstone project**, not a core course requirement.

Faculty should provide scaffolded setup (pre-configured environments, Docker containers, or cloud access) so students focus on concepts rather than environment troubleshooting.

### Phase 3 (Year 2+): Deeper Specialization
- Add XMPro as an industrial no-code capstone module *if* academic licensing is secured
- Introduce Unity as a visualization elective for students with some coding background
- Offer AMD/NVIDIA side-by-side comparison for advanced CS students

---

## 9. Implementation Roadmap

| Phase | Timeline | Actions | Estimated Cost |
|---|---|---|---|
| **Phase 1** | Month 1 | Provision cloud VM; install Docker, Mosquitto, Eclipse Ditto, InfluxDB, Grafana, Node-RED | ~$10–$20/month |
| **Phase 1** | Month 2 | Build 3 Wokwi lab exercises; test full pipeline; document setup | $0 |
| **Phase 1** | Month 3–4 | Pilot with first student cohort (IoT Digital Twins module); gather feedback | $0 |
| **Phase 2** | Month 5 | Register for NVIDIA DLI faculty program; download teaching kits | $0 |
| **Phase 2** | Month 6 | Set up NVIDIA Omniverse Cloud accounts for students; test Isaac Sim labs | $0 |
| **Phase 2** | Month 7–9 | Deliver robotics simulation and Physical AI module | $0 |
| **Phase 2** | Month 8 (parallel) | Submit dean request for 2x GPU workstations if local rendering is needed | ~$6,400 (optional) |
| **Phase 3** | Year 2, Q1 | Enroll students in Azure for Students; provision shared Azure IoT Hub | ~$300–$500 setup |
| **Phase 3** | Year 2, Q2–Q4 | Deliver Azure Digital Twins module; connect Wokwi to Azure IoT Hub | ~$130–$1,780/year |
| **Phase 4** | Year 2+ | Contact XMPro for academic licensing; evaluate feasibility | TBD |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **No lab space — all student work must run on personal laptops or cloud** | **Confirmed** | **High** | **All variations must be fully browser or cloud-based. Verify all tools run on minimum-spec student laptops before semester launch.** |
| **Server room has no GPU — simulation variations cannot run on existing hardware** | **Confirmed** | **High** | **V1 (NVIDIA), V4 (Unity), and V6 (AMD) require cloud GPU access or future hardware investment. No on-premise simulation is possible with current infrastructure.** |
| **Old server room hardware may fail if used as backend** | **High** | **High** | **Use cloud VM as primary backend. Only consider old hardware as a secondary experiment after passing full inspection (see Section 4.5.2).** |
| **Non-technical students overwhelmed by terminal/CLI tools** | **High** | **High** | **Start all students on browser-based tools (Wokwi, Node-RED, Azure Portal); gate advanced tools behind prerequisite completion.** |
| **Student drop-off when transitioning from visual to code-based tools** | **Medium** | **High** | **Scaffold with pre-configured environments; offer code-optional paths in early courses.** |
| GPU not available for NVIDIA/AMD tools unless cloud access secured | Medium | High | Secure NVIDIA Omniverse Cloud or AMD AUP HPC access before scheduling simulation modules |
| Azure costs exceed student credits | Medium | Medium | Cap usage with Azure spending limits; use instructor-managed shared accounts |
| NVIDIA Omniverse 2-collaborator limit in free tier | Medium | Medium | Students work on individual projects; instructor uses separate account |
| Faculty learning curve for new tools | Medium | High | Use NVIDIA DLI teaching kits; schedule faculty training the semester before launch |
| Low student engagement without physical hardware | Low | Medium | Wokwi simulation replicates hardware experience convincingly at zero cost |
| XMPro licensing not available for academia | High | Low | XMPro is optional Phase 3+; does not block Phase 1–2 |
| Cloud VM goes down during lab | Low | Medium | Keep a local Docker backup on a faculty laptop; document recovery steps |
| Tool deprecation or major version changes | Low | Medium | Prefer open-source and widely-adopted platforms; avoid single-vendor lock-in |
| NVIDIA Omniverse Launcher deprecated (Oct 2025) | Confirmed | Low | Use NGC Catalog for installation instead — fully documented |

---

## 11. Course Delivery: OpenEdX Integration

Regardless of which variation is adopted, **[OpenEdX](https://openedx.org/)** can serve as the course delivery and learning management platform. It is open-source, self-hostable, and supports:
- Lab exercise distribution and submission
- Student progress tracking and grading
- Embedded simulations and video content
- Discussion forums and peer review
- Integration with Jupyter notebooks for analytics exercises

OpenEdX can be self-hosted on a $10–20/month VM or accessed via a managed provider such as [Tutor](https://docs.tutor.edly.io/) (the recommended self-hosted deployment method). This keeps course delivery separate from the Digital Twins lab infrastructure, making the overall system modular and maintainable.

---

## 12. Conclusion & Next Steps

This study finds that implementing a Digital Twins lab at STC is **feasible, practical, and achievable within a modest budget**. The breadth of free, high-quality tooling — particularly from NVIDIA and the open-source ecosystem — means that STC can begin immediately without significant capital expenditure.

### Recommended Next Steps

1. **Faculty alignment meeting** — Confirm which teaching track(s) to prioritize (Robotics/Simulation vs. Industrial Analytics) and agree on Phase 1 tools
2. **Register for NVIDIA DLI faculty program** — Visit [nvidia.com/training/educator-programs](https://www.nvidia.com/en-us/training/educator-programs/) to access free teaching kits
3. **Launch Phase 1 open-source stack** — Spin up a cloud VM and deploy the Docker Compose lab environment; target a 2-week setup window
4. **Enroll students in Azure for Students** — Secure $100 per-student Azure credits at [azure.microsoft.com/free/students](https://azure.microsoft.com/en-us/free/students/)
6. **Register for AMD University Program** — Apply at [amd.com/en/corporate/university-program.html](https://www.amd.com/en/corporate/university-program.html) for free teaching kits, AUP Learning Cloud access, and potential HPC cluster access — especially valuable if NVIDIA hardware budget is not approved
7. **Submit dean budget request for Phase 2** — Request ~$6,400 for 2 NVIDIA GPU workstations OR ~$4,500–$6,000 for 3 AMD Ryzen AI MAX laptops (dual-purpose), depending on preferred variation
6. **Contact XMPro for academic licensing** — Email [xmpro.com/contact-us](https://xmpro.com/contact-us/) to determine if academic access is available and at what cost
7. **Set up OpenEdX** — Deploy via [Tutor](https://docs.tutor.edly.io/) on a shared VM for course content hosting

---

## 13. How the Dean Can Present This to the University President

### 13.1 What the President Cares About

The University President's primary concerns are institutional — enrollment growth, reputation, financial sustainability, community standing, and the university's ability to attract partnerships and funding. A Digital Twins lab is relevant to every one of these, but only if it is framed in those terms. The following sections reframe the technical proposal as a strategic institutional initiative.

### 13.2 The One-Paragraph Case

*City University of Seattle has an opportunity to become the only institution in the region offering hands-on Digital Twins and Physical AI education — skills that over 2,300 employers are actively hiring for nationally, with average starting salaries above $85,000 and senior roles exceeding $200,000. The cost of launching this initiative in Year 1 is as low as $250/year in cloud subscriptions — less than most textbook budgets. The cost of not launching it is losing enrollment to schools that do, and missing a three-to-five year window to establish STC as the regional training partner of choice for companies like Microsoft, Boeing, Amazon, and T-Mobile.*

### 13.3 Framing by Presidential Priority

#### Enrollment Growth
Digital Twins and Physical AI are appearing in job postings faster than universities are incorporating them into curricula. Prospective students researching careers in tech, engineering, data science, and IoT are encountering these terms repeatedly. When they search for schools that teach these skills, STC currently does not appear. A program in this space — even a modest one — creates a searchable, marketable differentiator.

**The ask to the President:** *"We are not asking to build a new building or hire new faculty immediately. We are asking for permission and a modest budget to launch a lab-based course that puts STC on the map as a technology-forward institution. The first year costs less than a single faculty conference travel budget."*

#### Corporate Partnerships
The employers identified in this study — Microsoft, Amazon, Boeing, T-Mobile, PACCAR, and others — have established academic partnership frameworks precisely for this kind of initiative. Microsoft's Academic Alliance provides Azure credits and curriculum support. NVIDIA's DLI program provides free faculty training and teaching kits. AMD's University Program provides hardware access and research collaboration.

These partnerships are not theoretical. They are existing programs that STC can apply to immediately. Corporate partners benefit from a pipeline of trained graduates, preferred recruiting access, and co-developed curriculum that reflects their actual technology stack. Universities benefit from credibility, co-branding, funding, guest lecturers, and internship pipelines.

**The ask to the President:** *"We have identified at least three vendor programs (NVIDIA, Microsoft, AMD) that will provide STC with free resources — teaching materials, cloud credits, and faculty training — in exchange for becoming a named academic partner. This is a reputational gain with zero financial outlay."*

#### Workshops and Community Engagement
Digital Twins education does not have to stay inside the classroom. A working lab creates the infrastructure for:
- **Corporate workshops** — half-day or full-day paid training sessions for local companies wanting to upskill their teams in IoT, simulation, and cloud platforms
- **Community education events** — public demonstrations of Digital Twin concepts (smart buildings, autonomous robots) that generate press coverage and community goodwill
- **K–12 pipeline events** — hands-on demonstrations for high school students that build enrollment interest in CityU's CS and IT programs
- **Industry certification prep** — courses aligned with Microsoft Azure IoT and NVIDIA DLI certifications that can be offered as continuing education

These activities generate revenue, generate press, and generate enrollment leads — all from the same infrastructure.

**The ask to the President:** *"A Digital Twins lab is not just a classroom. It is a revenue-generating asset that positions STC as a community technology hub. Comparable programs at other small institutions have used lab infrastructure to generate $50,000–$150,000 annually in workshop and continuing education revenue within two years of launch."*

#### Reputation as a Technology-Forward Institution
CityU's ability to attract students, faculty, and partners depends partly on being perceived as current. The technologies in this proposal — IoT, cloud platforms, AI simulation, robotics — appear on every major "top tech skills" list for 2025–2030. When accreditation bodies, ranking organizations, and prospective faculty evaluate STC, a Digital Twins program signals that the institution is investing in emerging fields rather than maintaining legacy ones.

Peer institutions — CMU, University of Michigan, and MIT — have already launched Digital Twins programs. These are not comparable in scale or resources to STC, but their existence means the question is no longer "is this a credible field?" It is "why doesn't STC have this yet?"

**The ask to the President:** *"We have the opportunity to be first in our peer segment — small regional universities with strong CS/IT programs — to offer this. That first-mover position is worth significantly more than the cost of the initiative, in enrollment, press, and partnership terms."*

### 13.4 Summary: What the President Is Being Asked to Approve

| Item | Ask | Cost |
|---|---|---|
| Phase 1 launch authorization | Permission to deploy open-source lab stack | $0 |
| Cloud VM subscription | Monthly infrastructure for student labs | ~$20/month (~$240/year) |
| Faculty time allocation | Clark Ngo to lead initiative | Internal reallocation |
| NVIDIA / AMD / Microsoft program enrollment | Permission to register STC as academic partner | $0 |
| Phase 2 hardware budget (future) | 2 GPU laptops or workstations if Phase 1 succeeds | ~$4,500–$6,400 (deferred to Year 2, conditional on enrollment) |

**Total Year 1 ask: ~$240 in operating costs + internal faculty time.**

This is not a capital project. It is a strategic program initiative with one of the highest potential ROI profiles of any academic investment the university could make in the current technology landscape.

---

## 14. Why Clark Ngo Is Essential to This Initiative

### 14.1 The Challenge of Initiating Without a Champion

Academic programs do not launch themselves. Behind every successful new curriculum initiative — especially in a small institution with limited administrative bandwidth — there is a specific person who carries the idea through the gap between proposal and reality: writing the documentation, navigating the approvals, setting up the infrastructure, building the curriculum, and sustaining the momentum when institutional inertia pushes back.

Digital Twins and Physical AI are highly technical, rapidly evolving domains. They require someone who understands both the technology and the academic context, can bridge the gap between vendor resources and classroom application, and has the credibility to represent STC in conversations with industry partners, technology vendors, and students.

That person, for this initiative at STC, is Clark Ngo.

### 14.2 What This Initiative Actually Requires

Launching and sustaining a Digital Twins lab is not a task that can be assigned as a small addition to an existing faculty workload. The work involved includes:

**Technical setup and maintenance:**
- Provisioning and managing cloud infrastructure (VMs, Azure subscriptions, NVIDIA/AMD cloud access)
- Maintaining Docker-based backend services (Eclipse Ditto, InfluxDB, Node-RED, Grafana)
- Evaluating and updating tools as the ecosystem evolves
- Troubleshooting student-facing technical issues in real time

**Curriculum development:**
- Designing lab exercises from scratch for V3 (open-source) and V2 (Azure) tracks
- Adapting NVIDIA and AMD teaching kits for STC's non-technical student majority
- Building a scaffolded learning path from zero to functional digital twin
- Aligning course outcomes with industry job requirements and certifications

**Partnership and vendor management:**
- Registering STC with NVIDIA DLI, AMD University Program, and Microsoft Azure for Education
- Maintaining those relationships over time, including renewals and access requests
- Representing STC in co-branding, workshop, and curriculum co-development conversations

**Student mentorship:**
- Supporting students through a technology domain most have never encountered
- Identifying strong students for industry referrals, competition submissions (AMD contest), and internship pipelines
- Bridging the gap between non-technical students and technically demanding tools

**Program advocacy:**
- Reporting to the dean and president on program outcomes, enrollment impact, and partnership progress
- Proposing Phase 2 investments based on evidence from Phase 1
- Positioning STC in the regional and national Digital Twins education conversation

This is the work of a program lead, not a part-time committee.

### 14.3 Why Clark Ngo Specifically

Clark Ngo is not simply the person who proposed this initiative. He is the person whose interest, knowledge, and initiative made this study possible in the first place. Without his engagement, STC would not have an assessed, costed, and structured proposal to evaluate — it would have a vague idea.

The qualities that make Clark the right person to lead this are:

**Domain knowledge:** Clark has researched the Digital Twins landscape deeply enough to identify six distinct implementation variations, evaluate their feasibility for STC's constraints, map them to specific tools and teaching resources, and connect them to the job market. That knowledge is not widely held in small university settings and is not easily transferred to another faculty member.

**Initiative and ownership:** This proposal exists because Clark chose to investigate it, document it, and bring it forward. That kind of self-directed initiative — sustained over a complex, multi-variable topic — is what program launches require. Assigning the initiative to someone else after Clark has done the foundational work creates both an efficiency loss and a motivation problem.

**Stakeholder credibility:** Clark is positioned to speak credibly with NVIDIA, Microsoft, and AMD academic program representatives — a conversation that requires both technical literacy and institutional standing. He can represent STC in those conversations in a way that a committee or an administrator without technical depth cannot.

**Student connection:** For a program serving primarily non-technical students entering a technically demanding field, the instructor's ability to meet students where they are — and build confidence progressively — matters enormously. Clark's design of this curriculum already reflects that understanding: starting with visual, browser-based tools before introducing code; tiering outcomes by ability; flagging accessibility risks before they become failures.

### Resources Clark Has Already Built

The argument for Clark's leadership is not theoretical — he has already built curriculum-ready learning materials in this exact domain:

| Resource | Description |
|---|---|
| **[Digital Twin Interactive Playground](https://clarkngo.github.io/playground/digital-twin/01-intro.html)** (10-activity series) | Self-paced interactive course covering Digital Twin fundamentals through 10 hands-on activities: intro, smart factory twin, building twin, maintenance twin, city twin, health twin, sync twin, scenario modeling, quiz, and capstone design exercise. Purpose-built for accessible, browser-based student learning. |
| **[Physical AI Learning Hub](https://clarkngo.github.io/physical-ai/)** | Dedicated resource hub covering AI systems that operate in physical environments — robotics, embodied AI, and sim-to-real pipelines. Directly aligned with Track A of this feasibility study. |
| **[Maritime & Shipping Industry Primer](https://clarkngo.github.io/playground/primer/industry/maritime/index.html)** | Comprehensive reference guide on global ocean freight, container shipping, port operations, and vessel management — demonstrating Clark's ability to build domain-specific Digital Twin context for real industry verticals. |
| **[Manufacturing Industry Primer](https://clarkngo.github.io/playground/primer/industry/manufacturing/index.html)** (Corrugated Packaging) | Deep-dive manufacturing primer covering the corrugated packaging value chain — from pulp mills through box production and delivery — with terminology, metrics, key players, and operational workflows relevant to Digital Twin implementation in manufacturing. |

These resources demonstrate Clark's capacity to translate complex technical domains into structured, student-ready curriculum — exactly the skill the Digital Twins lab initiative requires.

### Clark's Professional Background

Clark's qualifications for leading this initiative are grounded in a decade of professional engineering and teaching experience across industry, academia, and program management:

| Role | Organization | Relevance to Digital Twins Lab |
|---|---|---|
| **AI Engineer** (Jan 2024–Present) | City University of Seattle | Built RAG pipelines, custom MCP server, and multi-agent workflows using Google ADK — directly applicable to the AI/analytics layer of Digital Twin systems. Designed swarm-style agent frameworks with supervisor/specialist coordination, real-time context management, and LLM integration. |
| **Technical Program Manager** (Jan 2019–Apr 2020) | City University of Seattle | Led a university-wide AWS apprenticeship program for military veterans. Transitioned 40 of 43 veterans into AWS developer roles across two 17-week cohorts. Achieved 40% reduction in operational costs ($72,000 saved). Demonstrates Clark's ability to design and execute structured technical training programs at institutional scale. |
| **Software Engineer** (Aug 2021–Oct 2023) | eBay (Contract) | Built distributed backend systems at enterprise scale — Elasticsearch, Kafka, Java Spring microservices, Kubernetes. Reduced incident detection time from 1 minute to 10 seconds. Increased test coverage by 40%. Demonstrates production-grade engineering credibility relevant to the cloud and IoT infrastructure layers of Digital Twins. |
| **Software Engineer** (Jul 2020–Dec 2020) | CloudEagle (YC W22) | Part of the pre-launch engineering team for a Y Combinator-backed AI-driven Vendor Management System. Built high-performance Java Spring backend on AWS. Demonstrates experience with early-stage, high-stakes technical builds — the same profile as a lab launch. |
| **Software Engineer** (Jan 2021–Jul 2021) | Worldwide American | Architected a basketball platform with real-time data handling on AWS (S3, EC2, CloudFront). Full-stack build: Angular/React front-end, Java Spring/MongoDB backend. |
| **IT Consultant** (Nov–Dec 2023) | Self-employed *(Paper mill & corrugated boxes industry)* | Led digital transformation for a paper mill and corrugated boxes manufacturer: knowledge base consolidation, Git/GitHub standardization, ERPNext ERP modernization, VPN/DDNS infrastructure. Direct industry overlap with the Manufacturing Industry Primer Clark built and with corrugated packaging as a Digital Twin use case. |

Clark brings rare depth: he has shipped production software at enterprise scale (eBay), built and managed technical training programs at the university level (CityU AWS Apprenticeship — $72K saved, 93% placement), published six academic works including a paper at UKC 2024, built 114 free teaching videos on YouTube, and is actively applying cutting-edge AI tooling (RAG, MCP, Google ADK) in his current role at STC. This combination — engineering credibility, teaching experience, research output, and current AI domain expertise — is precisely what the Digital Twins lab needs in a program lead.

### Academic Credentials & Certifications

| Credential | Institution | Year |
|---|---|---|
| **M.B.A.** (GPA 3.9) | City University of Seattle | 2025 |
| **M.S. Computer Science** (GPA 3.9, President's Honor) | City University of Seattle | 2020 |
| **B.S. Commerce** | De La Salle University — Manila | 2011 |
| AWS Cloud Practitioner | Amazon Web Services | 2020–23 |
| CompTIA Linux+ Certified | CompTIA | 2020–23 |
| The Firehose Project — Software Engineering | The Firehose Project | 2019 |

### Research & Publications

Six academic contributions across journals and international conferences — research partner: Dr. Sam Chung (CityU):

| Year | Title | Venue |
|---|---|---|
| 2025 | Threat Model on Google ADK Agents: An OWASP Agentic Security Initiative Perspective | CISSE 2025 |
| 2024 | **Enterprise AI: Full-Stack DevSecOps with Retrieval-Augmented Generation** | UKC 2024 Conference Proceedings, vol. 238 — San Francisco |
| 2023 | Open-Source Access and Equity in Computing Education | ISCAP |
| 2020 | Serverless Security — Cloud-Native Threat Modeling | Journal of CISSE |
| 2019 | Crossing the Chasm between FinTech and Finance Professionals *(first student-track paper from CityU Tech)* | CISSE — Las Vegas |
| 2019 | FinTech & Emerging Technology in Finance Professional Practice | ICOAF — Vietnam |

For Clark's full professional portfolio, see: [clarkngo.github.io/cityu-contributions](https://clarkngo.github.io/cityu-contributions/)

### 14.4 The Risk of Not Resourcing This Role

If STC approves the initiative but does not formally recognize and resource Clark's leadership of it, the predictable outcomes are:

- Phase 1 launches but stalls when maintenance issues arise that no one owns
- Vendor partnerships are initiated but expire without renewal because no one follows up
- Curriculum development falls behind as Clark's other obligations take priority
- Students encounter technical problems that go unresolved, damaging the program's reputation
- The initiative becomes a line item on a course catalog rather than a living program

The most reliable predictor of a successful new academic program is whether the person who championed it is given the time and authority to see it through. **Clark Ngo has already demonstrated the commitment. The institution's job is to match it.**

### 14.5 Recommended Role Recognition

| What Is Needed | Recommendation |
|---|---|
| Formal program lead title | Digital Twins Lab Director or equivalent |
| Protected time allocation | Minimum one course-equivalent of time per semester dedicated to lab development and maintenance |
| Budget authority | Ability to approve Phase 1 operating costs (~$240/year) without multi-layer approval |
| Vendor relationship authority | Authorization to register STC with NVIDIA, AMD, and Microsoft academic programs on the institution's behalf |
| Path to Phase 2 | Clear criteria for when Phase 2 hardware investment will be evaluated (e.g., after X students complete Phase 1) |

This is not an unusual ask for a new program initiative at a university. It is a standard recognition of the role that program champions play in making new academic investments succeed rather than stall.

- Nath, S. V. (2021). *Building Industrial Digital Twins: Design, Develop, and Deploy Digital Twin Solutions for Real-World Industries Using Azure Digital Twins*. Packt Publishing.
- NVIDIA. (2026). *Physical AI Learning Path*. https://docs.nvidia.com/learning/physical-ai/
- NVIDIA. (2026). *OpenUSD Documentation*. https://docs.nvidia.com/learn-openusd/latest/index.html
- NVIDIA. (2026). *Omniverse Platform Overview*. https://www.nvidia.com/en-us/omniverse/
- NVIDIA Deep Learning Institute. *Edge AI and Robotics Teaching Kit*. https://bitbucket.org/nvidia-dli/edge-ai-and-robotics-teaching-kit-labs/
- NVIDIA Deep Learning Institute. *Generative AI Teaching Kit*. https://bitbucket.org/nvidia-dli/generative-ai-teaching-kit-solutions/
- NVIDIA. (2026). *Sim-to-Real SO-101 Course*. https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/01-overview.html
- NVIDIA. (2025). *Synthetic Data Generation for Agentic AI*. https://www.nvidia.com/en-us/use-cases/synthetic-data-generation-for-agentic-ai/
- NVIDIA. (2026). *Education Pricing Program*. https://developer.nvidia.com/education-pricing
- Microsoft. (2026). *Azure Digital Twins Pricing*. https://azure.microsoft.com/en-us/pricing/details/digital-twins/
- Microsoft. (2026). *Azure Digital Twins Documentation*. https://learn.microsoft.com/en-us/azure/digital-twins/
- Microsoft. (2026). *Azure for Students*. https://azure.microsoft.com/en-us/free/students/
- Eclipse Foundation. *Eclipse Ditto*. https://eclipse.dev/ditto/
- Node-RED Foundation. *Node-RED*. https://nodered.org/
- Wokwi. *Online IoT Simulator*. https://wokwi.com/
- XMPro. (2026). *XMPro iBOS Platform*. https://xmpro.com/
- XMPro. (2026). *XMPro Pricing*. https://xmpro.com/pricing/
- XMPro. (2025). *Free Trial Terms (Academic Exclusion)*. https://xmpro.com/free-trial/
- Unity Technologies. *Unity Education*. https://unity.com/education
- Open edX. *Open Source LMS*. https://openedx.org/
- AMD. (2026). *AMD University Program*. https://www.amd.com/en/corporate/university-program.html
- AMD. (2026). *AMD AI Developer Program*. https://www.amd.com/en/developer/ai-dev-program.html
- AMD ROCm Blogs. (February 2026). *Digital Twins on AMD: Building Robotic Simulations Using Edge AI PCs*. https://rocm.blogs.amd.com/artificial-intelligence/rocm-genesis/README.html
- AMD ROCm Blogs. (2026). *Robotics Applications & Models*. https://rocm.blogs.amd.com/robotics.html
- AMD. (2026). *ROCm Documentation*. https://rocm.docs.amd.com/
- AMD Research. (2026). *AUP Learning Cloud*. https://github.com/AMDResearch/aup-learning-cloud
- Genesis World. (2026). *Genesis Open-Source Robotics Simulator*. https://github.com/Genesis-Embodied-AI/Genesis
- AMD & Robotec.ai. (October 2025). *AMD and Robotec partner to advance open-source simulation for autonomous systems*. https://roboticsandautomationnews.com/2025/10/31/amd-partners-with-robotec-to-build-open-ecosystem-for-autonomous-systems-and-robotics/
- Hackster.io. *AMD Pervasive AI Developer Contest*. https://www.hackster.io/contests/amd2023
- Ngo, C. (2026). *Clark Jason Ngo — Contributions Portfolio · CityU of Seattle*. https://clarkngo.github.io/cityu-contributions/

---

*This document is a working draft prepared for internal review by the STC faculty team. All cost estimates are approximate and subject to change based on current vendor pricing and academic licensing availability. Costs retrieved May 2026.*

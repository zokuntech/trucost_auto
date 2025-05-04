# 📄 context.md — Car Repair Audit Platform (Backend Plan)

---

## 🎯 High-Level Architecture

- **Main Backend (FastAPI)**: Conductor that receives user requests, routes them to correct services.
- **Agents (Microservices):** Smart modules handling complex, fuzzy tasks.
- **APIs (Microservices):** Dumb, mechanical services handling simple tasks.

---

## 🛠️ Core Services Breakdown

| Service | Type | Purpose |
|:---|:---|:---|
| Upload API | Microservice/API | Handle file upload (PDF, image, or text) |
| Quote Parser Agent | Agent | Read messy quotes → extract parts, tasks, prices |
| Parts Research Agent | Agent | Search Amazon, RockAuto, AutoZone → find cheapest parts |
| Labor Lookup API | Microservice/API | Given task name → return fair labor hours + labor rate |
| Audit Comparison API | Microservice/API | Compare mechanic quote vs found parts/labor and calculate savings |
| Stripe Payment API | Microservice/API | Handle $5 payment after first free audit |
| Affiliate Link Fetcher API | Microservice/API | Fetch and embed affiliate links for found parts |


---

## 🧠 Flow of Operations

```plaintext
User ➔ FastAPI Backend
    ├─ Upload API (store file/text)
    ├─ Call ➔ Quote Parser Agent (extract parts/tasks)
    ├─ Call ➔ Parts Research Agent (find cheapest parts)
    ├─ Call ➔ Labor Lookup API (get fair labor times/costs)
    ├─ Call ➔ Audit Comparison API (calculate savings)
    └─ Return savings report JSON (includes affiliate links)
```

---

## 📦 Folder Structure Overview

```plaintext
/backend
  /api
    upload.py
    audit.py
    payment.py
  /agents
    /quote_parser
    /parts_research
  /services
    labor_lookup.py
    affiliate_links.py
  /models
    audit_models.py
    quote_models.py
  /utils
    vin_tools.py
    file_parser.py
  main.py
  requirements.txt
```

✅ FastAPI app = orchestrator.
✅ Agents = separate lightweight FastAPI microservices.
✅ All communications via internal API calls.


---

## 🔥 Key MVP Rules

- **Backend API is Conductor** — not an agent itself.
- **Agents only spun up where "thinking" is needed** (messy parsing, fuzzy parts matching).
- **APIs handle dumb tasks** like file upload, database CRUD, Stripe calls.
- **Supervisor Agent not needed for MVP.** Only upgrade to Supervisor if system complexity grows later.


---

# ⚡ Savage TL;DR:
> **Backend handles orchestration directly.**
>
> **2 smart agents (Quote Parser + Parts Research).**
>
> **Everything else is simple APIs.**
>
> **Simple, fast, scalable launch plan.**

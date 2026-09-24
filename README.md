final architecture:

Streamlit
   │
   ▼
CrewAI Sequential Crew
   │
   ├── Task 1
   │     └── Agent 1: RAG
   │             └── Local Knowledge Base
   │
   ├── Task 2
   │     └── Agent 2: Web Search
   │             └── Web Search Tool
   │
   └── Task 3
         └── Agent 3: Entry/Record
                 └── TXT File
   │
   ▼
Streamlit
   ├── RAG Answer
   ├── Web Search Answer
   └── Saved Record


   LLM = gpt-4o-mini
API key = .env
Crew process = Process.sequential
UI = Streamlit
Framework = CrewAI
Code = app.py only


---------------------------
Crew_Customer_Support/
│
├── app.py
├── .env
├── .gitignore
├── outputs/
└── venv/
-----------------------------------
                ┌──────────────────┐
                │  Streamlit UI    │
                │  User Query      │
                └────────┬─────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   CrewAI Crew         │
              │   Sequential Process │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Agent 1 — RAG        │
              │ Knowledge Retrieval  │
              └──────────┬───────────┘
                         │
                         │ RAG Answer
                         ▼
              ┌──────────────────────┐
              │ Agent 2 — Web Search │
              │ Internet Research    │
              └──────────┬───────────┘
                         │
                         │ Web Answer
                         ▼
              ┌──────────────────────┐
              │ Agent 3 — Entry      │
              │ Save TXT             │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Streamlit UI       │
              │ RAG Answer           │
              │ Web Answer           │
              │ Saved File            │
              └──────────────────────┘

              ------------------------------------------
              Agent 1:

Search local knowledge base
        ↓
Retrieve relevant chunks
        ↓
Generate answer using GPT-4o-mini

Agent 2 — Web Search Agent

Then Agent 2 gets the original question and Agent 1's result.

User Query
    +
RAG Answer
    ↓
Web Search
    ↓
Current/relevant web information
    ↓
Web Search Answer

This is particularly useful when the RAG knowledge base contains older/static information but the web may have newer information.

Agent 3 — Save TXT Agent

Finally:

Query
+
RAG Answer
+
Web Answer
        ↓
TXT file

For example:

outputs/
    support_20260924_183000.txt

Contents:

CUSTOMER SUPPORT QUERY
======================

Query:
What is CASA in banking?

RAG ANSWER
==========
...

WEB SEARCH ANSWER
=================
# Research Assistant Agent 🔍

An agentic AI system that autonomously searches the web and reasons over results
to answer research questions. Built with LangGraph, Groq LLM, and Tavily Search.

## Architecture
```
User Question
     ↓
[Researcher Node]  → Searches web using Tavily API
     ↓
[Reasoner Node]    → LLM evaluates results, forms answer
     ↓
  Sufficient?
  ↓ Yes          ↓ No (loops back, max 2 iterations)
[Responder Node]  [Researcher Node]
     ↓
Final Answer
```

## Features

- **Agentic Loop** — agent autonomously decides whether to search again or respond
- **ReAct Pattern** — Reason + Act loop with a configurable iteration limit
- **Real-time Web Search** — Tavily API for up-to-date information retrieval
- **LLM Reasoning** — Groq (LLaMA 3.1) evaluates result quality before responding
- **REST API** — FastAPI backend with Swagger UI for easy testing

## Tech Stack

Python · LangGraph · LangChain · Groq API (LLaMA 3.1) · Tavily Search · FastAPI · Pydantic

## Project Structure
```
research-assistant-agent/
├── app/
│   ├── agent.py    # LangGraph agent — state, nodes, edges, graph
│   ├── tools.py    # Tavily web search tool
│   └── main.py     # FastAPI endpoints
└── .env            # API keys (not committed)
```

## Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/sakthee30/research-assistant-agent.git
cd research-assistant-agent
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install langgraph langchain langchain-groq tavily-python fastapi uvicorn python-dotenv
```

**4. Set up API keys**
```bash
# Create .env file and add your keys
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

**5. Run the server**
```bash
uvicorn app.main:app --reload
```

**6. Test via Swagger UI**

Open `http://127.0.0.1:8000/docs` in your browser.

## Example

**Request:**
```json
{
  "question": "What are the latest developments in large language models in 2025?"
}
```

**Response:**
```json
{
  "question": "What are the latest developments in large language models in 2025?",
  "answer": "The latest developments include multimodal AI, autonomous agents, RLHF...",
  "iterations": 1
}
```

## How the Agent Works

1. **Researcher Node** — calls Tavily to fetch top 3 web results for the question
2. **Reasoner Node** — LLM reads results and generates an answer, then self-evaluates
   whether the information was sufficient (returns `SUFFICIENT` / `INSUFFICIENT`)
3. **Conditional Edge** — if insufficient and under 2 iterations, loops back to search again
4. **Responder Node** — finalises and returns the clean answer
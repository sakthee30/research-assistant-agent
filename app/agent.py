import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.tools import search_web

load_dotenv()

# ── 1. STATE ──────────────────────────────────────────────
# This dictionary is passed between every node.
# Each node reads from it and updates it.

class AgentState(TypedDict):
    question: str          # original user question
    search_results: str    # raw results from Tavily
    answer: str            # LLM's answer
    is_sufficient: bool    # did LLM find enough info?
    iterations: int        # loop counter (max 2)


# ── 2. LLM SETUP ──────────────────────────────────────────

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.3
)


# ── 3. NODES ──────────────────────────────────────────────
# Each node is just a function. Receives state, returns
# a dict of only the fields it wants to update.

def researcher_node(state: AgentState) -> dict:
    """Searches the web for the user's question."""
    print(f"🔍 Searching web... (iteration {state['iterations'] + 1})")
    
    results = search_web(state["question"])
    
    return {
        "search_results": results,
        "iterations": state["iterations"] + 1
    }


def reasoner_node(state: AgentState) -> dict:
    """LLM reads search results and decides if it has enough info."""
    print("🧠 Reasoning over results...")
    
    messages = [
        SystemMessage(content="""You are a research assistant. 
        Given a question and search results, provide a clear answer.
        At the end of your response, on a new line write either:
        SUFFICIENT or INSUFFICIENT
        based on whether the search results gave you enough information."""),
        
        HumanMessage(content=f"""Question: {state['question']}
        
Search Results:
{state['search_results']}

Provide your answer, then on the last line write SUFFICIENT or INSUFFICIENT.""")
    ]
    
    response = llm.invoke(messages)
    full_response = response.content
    
    # Check if LLM thinks results were good enough
    is_sufficient = "INSUFFICIENT" not in full_response.upper().split("\n")[-1]
    
    # Clean answer - remove the SUFFICIENT/INSUFFICIENT line
    answer_lines = full_response.strip().split("\n")
    clean_answer = "\n".join(
        line for line in answer_lines 
        if line.strip() not in ["SUFFICIENT", "INSUFFICIENT"]
    ).strip()
    
    return {
        "answer": clean_answer,
        "is_sufficient": is_sufficient
    }


def responder_node(state: AgentState) -> dict:
    """Final node — formats and returns the answer."""
    print("✅ Preparing final answer...")
    return {"answer": state["answer"]}


# ── 4. CONDITIONAL EDGE ───────────────────────────────────
# This function decides what happens after the reasoner.
# It's the "brain" of the loop.

def should_continue(state: AgentState) -> str:
    """
    Returns the name of the next node to go to.
    - If answer is good enough → go to responder
    - If not and under 2 iterations → search again
    - If hit limit → go to responder anyway
    """
    if state["is_sufficient"]:
        return "responder"
    
    if state["iterations"] >= 2:
        print("⚠️  Max iterations reached, returning best answer.")
        return "responder"
    
    return "researcher"  # loop back


# ── 5. BUILD THE GRAPH ────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("responder", responder_node)
    
    # Set entry point - always start here
    graph.set_entry_point("researcher")
    
    # Fixed edge: researcher always goes to reasoner
    graph.add_edge("researcher", "reasoner")
    
    # Conditional edge: reasoner decides what's next
    graph.add_conditional_edges(
        "reasoner",
        should_continue,
        {
            "researcher": "researcher",  # loop back
            "responder": "responder"     # finish
        }
    )
    
    # Fixed edge: responder always ends the graph
    graph.add_edge("responder", END)
    
    return graph.compile()


# Compile once and reuse
agent = build_agent()
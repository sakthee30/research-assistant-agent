import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    """Search the web using Tavily and return results as a string."""
    response = tavily_client.search(
        query=query,
        max_results=3,
        search_depth="basic"
    )
    
    results = []
    for item in response["results"]:
        results.append(f"Title: {item['title']}\nContent: {item['content']}\n")
    
    return "\n".join(results)
from typing import Optional
from tavily import AsyncTavilyClient

async def web_search(query: str) -> str:
    """Search the web for information when the agent needs external knowledge.
    Use this tool when you need to find current or factual information."""
    try:
        client = AsyncTavilyClient()  # Create client instance here
        search = await client.search(query, search_depth="advanced")
        results = []
        for result in search['results']:
            results.append(f"Source: {result['url']}\nTitle: {result['title']}\nContent: {result['content']}\n")
        return "\n---\n".join(results)
    except Exception as e:
        return f"Search failed: {str(e)}"
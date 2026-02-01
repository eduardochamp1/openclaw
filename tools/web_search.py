"""
OpenClaw - Web Search Tool
==========================
Web search capabilities using DuckDuckGo, Serper, or Tavily.
"""

import os
import asyncio
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod

import httpx
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    content: str | None = None  # Full page content if fetched


class BaseSearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute a search query."""
        pass


class DuckDuckGoSearch(BaseSearchProvider):
    """DuckDuckGo search provider (free, no API key needed)."""
    
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using DuckDuckGo."""
        
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, _search)
        
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", "")
            )
            for r in raw_results
        ]


class SerperSearch(BaseSearchProvider):
    """Serper.dev search provider (requires API key)."""
    
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable required")
        self.base_url = "https://google.serper.dev/search"
    
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using Serper."""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={"X-API-KEY": self.api_key},
                json={"q": query, "num": max_results}
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for r in data.get("organic", [])[:max_results]:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", "")
            ))
        
        return results


class TavilySearch(BaseSearchProvider):
    """Tavily search provider (requires API key)."""
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable required")
        self.base_url = "https://api.tavily.com/search"
    
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using Tavily."""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True
                }
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for r in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")
            ))
        
        return results


class WebFetcher:
    """Fetches and extracts content from web pages."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    async def fetch(self, url: str) -> str | None:
        """Fetch and extract text content from a URL."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers, follow_redirects=True)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "lxml")
                
                # Remove unwanted elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                
                # Extract text
                text = soup.get_text(separator="\n", strip=True)
                
                # Clean up excessive whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                return "\n".join(lines)
                
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to fetch {url}: {e}[/yellow]")
            return None


class WebSearchTool:
    """
    Main web search tool that combines search and content fetching.
    
    This is the tool exposed to the agent.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.max_results = config.get("max_results", 10)
        self.provider_name = config.get("provider", "duckduckgo")
        self.fetcher = WebFetcher()
        
        # Initialize search provider
        self.provider = self._get_provider()
    
    def _get_provider(self) -> BaseSearchProvider:
        """Get the configured search provider."""
        providers = {
            "duckduckgo": DuckDuckGoSearch,
            "serper": SerperSearch,
            "tavily": TavilySearch,
        }
        
        provider_class = providers.get(self.provider_name)
        if not provider_class:
            console.print(f"[yellow]Unknown provider {self.provider_name}, using DuckDuckGo[/yellow]")
            return DuckDuckGoSearch()
        
        try:
            return provider_class()
        except ValueError as e:
            console.print(f"[yellow]{e}, falling back to DuckDuckGo[/yellow]")
            return DuckDuckGoSearch()
    
    @property
    def definition(self) -> dict:
        """Return the tool definition for LLM."""
        return {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this when you need "
                "up-to-date information, facts, news, or data not in your training. "
                "Returns search results with titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and concise."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5, max: 10)",
                        "default": 5
                    },
                    "fetch_content": {
                        "type": "boolean",
                        "description": "Whether to fetch full page content for top results",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        }
    
    async def execute(
        self,
        query: str,
        max_results: int = 5,
        fetch_content: bool = False
    ) -> dict[str, Any]:
        """
        Execute a web search.
        
        Args:
            query: Search query string
            max_results: Maximum results to return
            fetch_content: Whether to fetch full page content
        
        Returns:
            Dictionary with search results
        """
        try:
            max_results = min(max_results, self.max_results)
            
            console.print(f"[blue]🔍 Searching: {query}[/blue]")
            results = await self.provider.search(query, max_results)
            
            # Optionally fetch full content for top results
            if fetch_content and results:
                console.print("[blue]📄 Fetching page content...[/blue]")
                # Fetch top 3 results
                for result in results[:3]:
                    content = await self.fetcher.fetch(result.url)
                    if content:
                        # Truncate content to avoid token limits
                        result.content = content[:5000]
            
            # Format results
            formatted_results = []
            for i, r in enumerate(results, 1):
                result_dict = {
                    "index": i,
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet
                }
                if r.content:
                    result_dict["content"] = r.content
                formatted_results.append(result_dict)
            
            return {
                "success": True,
                "query": query,
                "num_results": len(formatted_results),
                "results": formatted_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

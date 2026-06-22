"""
FastMCP server exposing IDP RAG capabilities as MCP tools.

Run:
  python -m mcp_server.server

Cursor MCP URL:
  http://localhost:8001/sse
"""

from typing import Any

import httpx
from fastmcp import FastMCP

from app.core.config import settings

mcp = FastMCP(
    name="IDP RAG Server",
    instructions="""
You have access to an Intelligent Document Processing (IDP) system.
Users can upload PDFs, then search and ask questions.
Always cite page numbers when answering.
""",
)

API_BASE = "http://localhost:8000"


@mcp.tool()
async def ask_documents(question: str, document_ids: list[str]) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_BASE}/query",
            json={"question": question, "document_ids": document_ids},
            timeout=120.0,
        )
        if resp.status_code != 200:
            return {"error": resp.text}
        return resp.json()


@mcp.tool()
async def list_documents() -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/documents", timeout=30.0)
        if resp.status_code != 200:
            return {"error": resp.text}
        return {"documents": resp.json()}


@mcp.tool()
async def get_document_status(document_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/documents/{document_id}/status", timeout=30.0)
        if resp.status_code != 200:
            return {"error": resp.text}
        return resp.json()


@mcp.tool()
async def get_document_entities(document_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/entities/{document_id}", timeout=30.0)
        if resp.status_code != 200:
            return {"error": resp.text}
        return resp.json()


if __name__ == "__main__":
    mcp.run(transport="sse", host=settings.mcp_server_host, port=settings.mcp_server_port)


"""HTTP client for the IDP RAG FastAPI backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import httpx

from app.core.config import settings

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _mime_for_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return _MIME_TYPES.get(ext, "application/octet-stream")


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


class IDPClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or settings.api_public_url).rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _handle(self, response: httpx.Response) -> Any:
        if response.is_success:
            if response.status_code == 204:
                return None
            return response.json()
        try:
            body = response.json()
            detail = body.get("error") or body.get("detail") or response.text
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail)
        except Exception:
            detail = response.text
        raise APIError(response.status_code, str(detail))

    def health(self) -> dict:
        with httpx.Client(timeout=10.0) as client:
            return self._handle(client.get(self._url("/health")))

    def list_documents(self) -> list[dict]:
        with httpx.Client(timeout=self.timeout) as client:
            return self._handle(client.get(self._url("/documents")))

    def upload_document(self, filename: str, file_bytes: bytes) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._url("/documents/upload"),
                files={"file": (filename, file_bytes, _mime_for_filename(filename))},
            )
            return self._handle(response)

    def document_status(self, document_id: str) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            return self._handle(client.get(self._url(f"/documents/{document_id}/status")))

    def delete_document(self, document_id: str) -> None:
        with httpx.Client(timeout=self.timeout) as client:
            self._handle(client.delete(self._url(f"/documents/{document_id}")))

    def query(
        self,
        question: str,
        document_ids: list[str],
        session_id: str | None = None,
        chat_history: list[dict] | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "question": question,
            "document_ids": document_ids,
        }
        if session_id:
            payload["session_id"] = session_id
        if chat_history:
            payload["chat_history"] = chat_history

        with httpx.Client(timeout=300.0) as client:
            response = client.post(self._url("/query"), json=payload)
            return self._handle(response)

    def query_stream(
        self,
        question: str,
        document_ids: list[str],
        session_id: str | None = None,
        chat_history: list[dict] | None = None,
    ) -> Iterator[dict]:
        payload: dict[str, Any] = {
            "question": question,
            "document_ids": document_ids,
        }
        if session_id:
            payload["session_id"] = session_id
        if chat_history:
            payload["chat_history"] = chat_history
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", self._url("/query/stream"), json=payload) as response:
                if not response.is_success:
                    text = response.read().decode()
                    raise APIError(response.status_code, text)
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        yield json.loads(line[6:])

    def get_entities(self, document_id: str) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            return self._handle(client.get(self._url(f"/entities/{document_id}")))

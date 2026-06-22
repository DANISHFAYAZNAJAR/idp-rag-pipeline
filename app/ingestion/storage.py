import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import aiofiles
import structlog

from app.core.config import settings

log = structlog.get_logger()


class BaseStorageService(ABC):
    @abstractmethod
    async def save(self, file_bytes: bytes, user_id: str, filename: str) -> str:
        """Save file and return its storage path."""

    @abstractmethod
    async def get(self, path: str) -> bytes:
        """Retrieve file bytes by path."""

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete file by path."""

    @abstractmethod
    def get_absolute_path(self, path: str) -> Path:
        """Return absolute Path for local access."""


class LocalStorageService(BaseStorageService):
    def __init__(self, base_path: str = settings.local_storage_path):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    async def save(self, file_bytes: bytes, user_id: str, filename: str) -> str:
        safe_name = f"{uuid.uuid4()}_{Path(filename).name}"
        user_dir = self.base / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        dest = user_dir / safe_name
        async with aiofiles.open(dest, "wb") as f:
            await f.write(file_bytes)
        rel_path = str(dest.relative_to(self.base))
        log.info("storage.saved", path=rel_path, size_bytes=len(file_bytes))
        return rel_path

    async def get(self, path: str) -> bytes:
        full = self.base / path
        async with aiofiles.open(full, "rb") as f:
            return await f.read()

    async def delete(self, path: str) -> None:
        full = self.base / path
        if full.exists():
            full.unlink()
            log.info("storage.deleted", path=path)

    def get_absolute_path(self, path: str) -> Path:
        return self.base / path


def get_storage_service() -> BaseStorageService:
    if settings.storage_backend == "local":
        return LocalStorageService()
    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")


storage_service = get_storage_service()

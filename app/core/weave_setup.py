"""Optional Weave/W&B tracing — never block app startup."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import structlog

from app.core.config import settings

log = structlog.get_logger()

_PLACEHOLDER_ENTITIES = {
    "",
    "your-wandb-username",
    "your_wandb_username",
    "changeme",
}


def weave_enabled() -> bool:
    if not settings.enable_weave:
        return False
    if not settings.wandb_api_key:
        return False
    entity = (settings.wandb_entity or "").strip().lower()
    if entity in _PLACEHOLDER_ENTITIES:
        return False
    return True


def init_weave() -> None:
    if not weave_enabled():
        log.info(
            "weave.disabled",
            enable_weave=settings.enable_weave,
            entity=settings.wandb_entity or "(unset)",
        )
        return

    import weave

    entity = settings.wandb_entity.strip()
    project = settings.wandb_project
    project_path = f"{entity}/{project}" if entity else project

    try:
        weave.init(project_path)
        log.info("weave.initialized", project=project_path)
    except Exception as exc:
        log.warning("weave.init_failed", error=str(exc), project=project_path)


@contextmanager
def weave_attributes(attrs: dict[str, Any]) -> Iterator[None]:
    if not weave_enabled():
        yield
        return

    import weave

    with weave.attributes(attrs):
        yield

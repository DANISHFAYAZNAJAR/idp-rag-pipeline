"""Sync SQLAlchemy engine for Celery worker background threads.

Async engine connections are bound to an event loop; worker threads must use
sync sessions (psycopg2) to avoid 'Future attached to a different loop' errors.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

sync_engine = create_engine(
    settings.database_sync_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False)

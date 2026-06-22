"""Block until Postgres DNS resolves and accepts connections (Docker entrypoint)."""

from __future__ import annotations

import os
import socket
import sys
import time
from urllib.parse import urlparse

import psycopg2


def _postgres_host() -> str:
    explicit = os.environ.get("POSTGRES_HOST")
    if explicit:
        return explicit
    url = os.environ.get("DATABASE_SYNC_URL", "")
    parsed = urlparse(url)
    return parsed.hostname or "postgres"


def _wait_for_dns(host: str, deadline: float) -> None:
    while time.time() < deadline:
        try:
            socket.getaddrinfo(host, None)
            print(f"Resolved Postgres host: {host}")
            return
        except socket.gaierror as exc:
            print(f"Waiting for DNS ({host}): {exc}", flush=True)
            time.sleep(2)
    raise TimeoutError(f"Could not resolve Postgres host '{host}'")


def main() -> None:
    url = os.environ.get("DATABASE_SYNC_URL")
    if not url:
        print("DATABASE_SYNC_URL is not set", file=sys.stderr)
        sys.exit(1)

    host = _postgres_host()
    timeout = int(os.environ.get("DB_WAIT_TIMEOUT", "120"))
    deadline = time.time() + timeout
    last_error = ""

    try:
        _wait_for_dns(host, deadline)
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    while time.time() < deadline:
        try:
            conn = psycopg2.connect(url)
            conn.close()
            print("Postgres is ready")
            return
        except psycopg2.OperationalError as exc:
            last_error = str(exc)
            time.sleep(2)

    print(f"Timed out waiting for Postgres: {last_error}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()

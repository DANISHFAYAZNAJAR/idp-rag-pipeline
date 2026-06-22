"""Start Celery ingestion worker with solo pool (required on macOS — prefork SIGSEGVs)."""


def main() -> None:
    from worker.celery_app import celery_app

    celery_app.worker_main(
        argv=[
            "worker",
            "-Q",
            "ingestion",
            "--loglevel=info",
            "--pool=solo",
            "--concurrency=1",
        ]
    )


if __name__ == "__main__":
    main()

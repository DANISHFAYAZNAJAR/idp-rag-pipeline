from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_chat_model_answer: str = "gpt-4o-mini"

    # W&B (required)
    wandb_api_key: str
    wandb_project: str = "idp-rag"
    wandb_entity: str = ""

    # Database
    database_url: str
    database_sync_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_store"

    # Storage
    storage_backend: str = "local"
    local_storage_path: str = "./uploads"

    # Auth
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    auth_enabled: bool = False

    # App
    environment: str = "development"
    log_level: str = "INFO"
    log_file: str = "./logs/app.jsonl"
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5

    # Rate limiting
    rate_limit_uploads_per_hour: int = 10
    rate_limit_queries_per_hour: int = 100

    # Feature flags
    enable_metrics: bool = False
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # MCP
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8001

    # Ingestion / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_chunks_per_doc: int = 2000
    embedding_batch_size: int = 50
    retrieval_top_k: int = 20
    reranker_top_n: int = 5
    confidence_threshold: float = -2.0
    ner_enabled: bool = True
    ner_batch_size: int = 5
    ner_concurrency: int = 8

    @staticmethod
    def user_collection_name(user_id: str) -> str:
        """One Chroma collection per user."""
        return f"user_{user_id.replace('-', '')}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "GraphRAG Knowledge Intelligence")
    environment: str = os.getenv("ENVIRONMENT", "development")
    database_path: str = os.getenv("DATABASE_PATH", "./data/graphrag.sqlite3")
    answer_provider: str = os.getenv("ANSWER_PROVIDER", "mock").lower()
    extraction_provider: str = os.getenv("EXTRACTION_PROVIDER", "heuristic").lower()
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "none").lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "160"))

    def ensure_data_dir(self) -> None:
        Path(self.database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


settings = Settings()

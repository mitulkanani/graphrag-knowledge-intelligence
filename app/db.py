import sqlite3
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT, content TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  model TEXT NOT NULL, vector_json TEXT NOT NULL, dimensions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, normalized_name TEXT NOT NULL UNIQUE,
  entity_type TEXT NOT NULL DEFAULT 'CONCEPT', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
  id TEXT PRIMARY KEY, source_entity_id TEXT NOT NULL REFERENCES entities(id),
  target_entity_id TEXT NOT NULL REFERENCES entities(id), relation_type TEXT NOT NULL,
  description TEXT NOT NULL, source_chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  confidence REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_entity_id);
CREATE TABLE IF NOT EXISTS chunk_entities (
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  PRIMARY KEY(chunk_id, entity_id)
);
CREATE TABLE IF NOT EXISTS communities (
  id TEXT PRIMARY KEY, label TEXT NOT NULL, summary TEXT NOT NULL,
  entity_count INTEGER NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS community_entities (
  community_id TEXT NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  PRIMARY KEY(community_id, entity_id)
);
CREATE TABLE IF NOT EXISTS query_runs (
  id TEXT PRIMARY KEY, question TEXT NOT NULL, mode TEXT NOT NULL,
  answer TEXT NOT NULL, citations_json TEXT NOT NULL, paths_json TEXT NOT NULL,
  debug_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_query_runs_created ON query_runs(created_at DESC);
"""

class Database:
    def __init__(self, path: str):
        self._uri = False
        self._anchor = None
        if path == ":memory:":
            # Keep an anchor connection open so short-lived sessions share one
            # in-memory database instead of each receiving a separate database.
            self.path = f"file:graphrag-{uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
            self._anchor = sqlite3.connect(self.path, uri=True, check_same_thread=False)
        else:
            self.path = path
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False, uri=self._uri)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def initialize(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def session(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

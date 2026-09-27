import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional
from app.core.config import settings
from app.core.logging import logger

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    tile_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    top_2_class TEXT NOT NULL DEFAULT '',
    top_2_confidence REAL NOT NULL DEFAULT 0.0,
    confidence_margin REAL NOT NULL DEFAULT 0.0,
    review_priority REAL NOT NULL DEFAULT 0.0,
    review_priority_level TEXT NOT NULL DEFAULT 'LOW',
    review_reason TEXT NOT NULL DEFAULT 'STANDARD_CONFIDENCE',
    status TEXT NOT NULL,
    all_probabilities TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_checksum TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    inference_latency_ms REAL NOT NULL,
    total_latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewed_class TEXT,
    comment TEXT,
    reviewer_id TEXT DEFAULT 'analyst_offline',
    reviewed_at TEXT NOT NULL,
    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_predictions_tile_hash ON predictions(tile_hash);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_class ON predictions(predicted_class);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_review_priority ON predictions(review_priority);
CREATE INDEX IF NOT EXISTS idx_predictions_priority_level ON predictions(review_priority_level);
CREATE INDEX IF NOT EXISTS idx_reviews_prediction_id ON prediction_reviews(prediction_id);
"""

class Database:
    """
    SQLite Database Manager configured for offline reliability, concurrency, and integrity.
    Uses Write-Ahead Logging (WAL) mode and busy timeout to ensure resilient concurrent reads/writes.
    """
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self._memory_conn: Optional[sqlite3.Connection] = None
        self._ensure_dir()
        
    def _ensure_dir(self) -> None:
        if self.db_path != ":memory:" and not self.db_path.startswith("file:"):
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn

        conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        
        # Optimize SQLite for reliable offline service
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def init_db(self) -> None:
        """Initializes database schema, applies non-destructive column migrations, and creates indexes."""
        logger.info(f"Initializing database at: {self.db_path}", extra={"event": "db_init"})
        with self.session() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            
            # Non-destructive migrations for existing tables
            cursor = conn.execute("PRAGMA table_info(predictions);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            
            migrations = [
                ("top_2_class", "TEXT NOT NULL DEFAULT ''"),
                ("top_2_confidence", "REAL NOT NULL DEFAULT 0.0"),
                ("confidence_margin", "REAL NOT NULL DEFAULT 0.0"),
                ("review_priority", "REAL NOT NULL DEFAULT 0.0"),
                ("review_priority_level", "TEXT NOT NULL DEFAULT 'LOW'"),
                ("review_reason", "TEXT NOT NULL DEFAULT 'STANDARD_CONFIDENCE'")
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_def};")

            # Create indexes after columns exist
            conn.executescript(CREATE_INDEXES_SQL)

        logger.info("Database schema and indexes successfully verified.", extra={"event": "db_ready"})

db = Database()

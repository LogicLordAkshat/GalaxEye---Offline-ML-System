import json
import sqlite3
from typing import Optional, List, Tuple
from app.db.database import db
from app.db.models import PredictionRecord, ReviewRecord
from app.core.logging import logger

class PredictionRepository:
    """
    Data access repository for prediction records and analyst reviews in SQLite.
    Abstracts persistence operations from domain/API layers.
    """
    def __init__(self, database=db):
        self.db = database

    def save(self, record: PredictionRecord) -> PredictionRecord:
        sql = """
        INSERT INTO predictions (
            prediction_id, tile_hash, filename, predicted_class, confidence,
            top_2_class, top_2_confidence, confidence_margin,
            review_priority, review_priority_level, review_reason,
            status, all_probabilities, model_name, model_version, model_checksum,
            preprocessing_version, inference_latency_ms, total_latency_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            record.prediction_id,
            record.tile_hash,
            record.filename,
            record.predicted_class,
            record.confidence,
            record.top_2_class,
            record.top_2_confidence,
            record.confidence_margin,
            record.review_priority,
            record.review_priority_level,
            record.review_reason,
            record.status,
            record.all_probabilities,
            record.model_name,
            record.model_version,
            record.model_checksum,
            record.preprocessing_version,
            record.inference_latency_ms,
            record.total_latency_ms,
            record.created_at
        )
        with self.db.session() as conn:
            cursor = conn.execute(sql, params)
            record.id = cursor.lastrowid
        return record

    def get_by_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        sql = "SELECT * FROM predictions WHERE prediction_id = ?;"
        with self.db.session() as conn:
            cursor = conn.execute(sql, (prediction_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_by_tile_hash(self, tile_hash: str) -> Optional[PredictionRecord]:
        """Finds the most recent prediction for a given tile hash (for duplicate caching)."""
        sql = "SELECT * FROM predictions WHERE tile_hash = ? ORDER BY id DESC LIMIT 1;"
        with self.db.session() as conn:
            cursor = conn.execute(sql, (tile_hash,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def query(
        self,
        class_name: Optional[str] = None,
        status: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_priority: Optional[float] = None,
        priority_level: Optional[str] = None,
        model_version: Optional[str] = None,
        sort_by: Optional[str] = "id",
        sort_order: Optional[str] = "desc",
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[PredictionRecord], int]:
        """
        Queries predictions matching optional filters with sorting and pagination.
        """
        conditions = []
        params: list = []

        if class_name:
            conditions.append("predicted_class = ?")
            params.append(class_name)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if min_priority is not None:
            conditions.append("review_priority >= ?")
            params.append(min_priority)
        if priority_level:
            conditions.append("review_priority_level = ?")
            params.append(priority_level)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Validate sorting column to prevent SQL injection
        allowed_sorts = {
            "id": "id",
            "review_priority": "review_priority",
            "confidence": "confidence",
            "confidence_margin": "confidence_margin",
            "created_at": "created_at"
        }
        sort_col = allowed_sorts.get(sort_by, "id")
        order_dir = "ASC" if (sort_order or "").lower() == "asc" else "DESC"

        count_sql = f"SELECT COUNT(*) FROM predictions {where_clause};"
        query_sql = f"""
        SELECT * FROM predictions 
        {where_clause} 
        ORDER BY {sort_col} {order_dir} 
        LIMIT ? OFFSET ?;
        """

        with self.db.session() as conn:
            total = conn.execute(count_sql, params).fetchone()[0]
            cursor = conn.execute(query_sql, params + [limit, offset])
            rows = cursor.fetchall()
            records = [self._row_to_record(r) for r in rows]

        return records, total

    def shortlist(
        self,
        mode: str = "needs_review",
        class_name: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        min_priority: Optional[float] = None,
        max_margin: float = 0.15,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100
    ) -> Tuple[List[PredictionRecord], int, str]:
        """
        Specialized analyst shortlisting and triage query.

        Modes:
        1. 'needs_review'      — Uncertain or high-priority tiles sorted by priority DESC.
        2. 'high_confidence'   — Auto-approvable (confidence >= 0.85, ACCEPTED), sorted by confidence DESC.
        3. 'conflict_margin'   — Competing top-2 classes (confidence_margin <= max_margin), sorted by margin ASC.
        4. 'critical_priority' — CRITICAL triage level tiles, sorted by priority DESC.
        5. 'per_class_anomaly' — Within each predicted class, tiles whose confidence is statistical outliers
                                  (>1.5 std below the class mean confidence). Surfaces likely mis-classified tiles.

        Optional filters: class_name, date_from (ISO 8601), date_to (ISO 8601).
        """
        where_parts: List[str] = []
        params: list = []
        order_by = "review_priority DESC"
        rule_desc = ""

        if class_name:
            where_parts.append("predicted_class = ?")
            params.append(class_name)

        if date_from:
            where_parts.append("created_at >= ?")
            params.append(date_from)

        if date_to:
            where_parts.append("created_at <= ?")
            params.append(date_to)

        if mode == "needs_review":
            max_c = max_confidence if max_confidence is not None else 0.70
            min_p = min_priority if min_priority is not None else 0.40
            rule_desc = f"Tiles requiring human review (Confidence ≤ {max_c:.2f} or Review Priority ≥ {min_p:.2f})"
            where_parts.append("(status = 'UNCERTAIN' OR confidence <= ? OR review_priority >= ?)")
            params.extend([max_c, min_p])
            order_by = "review_priority DESC, confidence_margin ASC"

        elif mode == "high_confidence":
            min_c = min_confidence if min_confidence is not None else 0.85
            rule_desc = f"Auto-approvable high-certainty predictions (Confidence ≥ {min_c:.2f})"
            where_parts.append("confidence >= ? AND status = 'ACCEPTED'")
            params.append(min_c)
            order_by = "confidence DESC"

        elif mode == "conflict_margin":
            rule_desc = f"Ambiguous predictions — top-2 class margin ≤ {max_margin:.2f}"
            where_parts.append("confidence_margin <= ?")
            params.append(max_margin)
            order_by = "confidence_margin ASC, review_priority DESC"

        elif mode == "critical_priority":
            rule_desc = "Triage priority CRITICAL (Priority Score ≥ 0.70)"
            where_parts.append("review_priority_level = 'CRITICAL'")
            order_by = "review_priority DESC"

        elif mode == "per_class_anomaly":
            # Statistical outlier detection: tiles with confidence > 1.5σ below their class mean.
            # Uses a SQLite subquery joining per-class stats to flag likely mis-classified tiles.
            rule_desc = "Per-class statistical anomalies — confidence > 1.5σ below class mean (potential mis-classifications)"
            order_by = "predicted_class ASC, confidence ASC"
            base_where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
            anomaly_sql = f"""
                SELECT p.*
                FROM predictions p
                JOIN (
                    SELECT predicted_class,
                           AVG(confidence) AS avg_conf,
                           MAX(0.0, AVG(confidence) - 1.5 * AVG(ABS(confidence - (
                               SELECT AVG(confidence) FROM predictions p2
                               WHERE p2.predicted_class = predictions.predicted_class
                           )))) AS threshold
                    FROM predictions
                    {base_where}
                    GROUP BY predicted_class
                    HAVING COUNT(*) >= 3
                ) stats ON p.predicted_class = stats.predicted_class
                    AND p.confidence < stats.threshold
                {('AND ' + ' AND '.join(where_parts)) if where_parts else ''}
                ORDER BY {order_by}
                LIMIT ?;
            """
            count_sql = f"""
                SELECT COUNT(*) FROM predictions p
                JOIN (
                    SELECT predicted_class,
                           MAX(0.0, AVG(confidence) - 1.5 * AVG(ABS(confidence - (
                               SELECT AVG(confidence) FROM predictions p2
                               WHERE p2.predicted_class = predictions.predicted_class
                           )))) AS threshold
                    FROM predictions
                    {base_where}
                    GROUP BY predicted_class
                    HAVING COUNT(*) >= 3
                ) stats ON p.predicted_class = stats.predicted_class
                    AND p.confidence < stats.threshold
                {('WHERE ' + ' AND '.join(where_parts)) if where_parts else ''};
            """
            with self.db.session() as conn:
                total = conn.execute(count_sql, params * 2 if where_parts else params).fetchone()[0]
                cursor = conn.execute(anomaly_sql, params * 2 + [limit] if where_parts else params + [limit])
                rows = cursor.fetchall()
                records = [self._row_to_record(r) for r in rows]
            return records, total, rule_desc

        else:
            rule_desc = "Standard prediction list"
            order_by = "id DESC"

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        count_sql = f"SELECT COUNT(*) FROM predictions {where_clause};"
        query_sql = f"SELECT * FROM predictions {where_clause} ORDER BY {order_by} LIMIT ?;"

        with self.db.session() as conn:
            total = conn.execute(count_sql, params).fetchone()[0]
            cursor = conn.execute(query_sql, params + [limit])
            rows = cursor.fetchall()
            records = [self._row_to_record(r) for r in rows]

        return records, total, rule_desc

    def save_review(self, review: ReviewRecord) -> ReviewRecord:
        """Saves a human analyst triage review/feedback record."""
        sql = """
        INSERT INTO prediction_reviews (
            prediction_id, decision, reviewed_class, comment, reviewer_id, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?);
        """
        params = (
            review.prediction_id,
            review.decision,
            review.reviewed_class,
            review.comment,
            review.reviewer_id,
            review.reviewed_at
        )
        with self.db.session() as conn:
            cursor = conn.execute(sql, params)
            review.id = cursor.lastrowid
        return review

    def get_reviews_for_prediction(self, prediction_id: str) -> List[ReviewRecord]:
        """Fetches all human reviews for a specific prediction."""
        sql = "SELECT * FROM prediction_reviews WHERE prediction_id = ? ORDER BY id DESC;"
        with self.db.session() as conn:
            cursor = conn.execute(sql, (prediction_id,))
            rows = cursor.fetchall()
            return [
                ReviewRecord(
                    id=r["id"],
                    prediction_id=r["prediction_id"],
                    decision=r["decision"],
                    reviewed_class=r["reviewed_class"],
                    comment=r["comment"],
                    reviewer_id=r["reviewer_id"],
                    reviewed_at=r["reviewed_at"]
                )
                for r in rows
            ]

    def get_summary_statistics(self) -> dict:
        """
        Aggregates summary statistics for offline observability and auditing.
        """
        with self.db.session() as conn:
            total_count = conn.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
            
            if total_count == 0:
                return {
                    "total_predictions": 0,
                    "status_distribution": [],
                    "priority_distribution": [],
                    "class_distribution": [],
                    "average_confidence": 0.0,
                    "average_review_priority": 0.0,
                    "average_inference_latency_ms": 0.0
                }

            # Status distribution
            status_rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM predictions GROUP BY status;"
            ).fetchall()
            status_dist = [
                {"status": r["status"], "count": r["cnt"], "percentage": round((r["cnt"] / total_count) * 100, 2)}
                for r in status_rows
            ]

            # Priority level distribution
            priority_rows = conn.execute(
                "SELECT review_priority_level, COUNT(*) as cnt FROM predictions GROUP BY review_priority_level;"
            ).fetchall()
            priority_dist = [
                {"level": r["review_priority_level"], "count": r["cnt"], "percentage": round((r["cnt"] / total_count) * 100, 2)}
                for r in priority_rows
            ]

            # Class distribution
            class_rows = conn.execute(
                "SELECT predicted_class, COUNT(*) as cnt FROM predictions GROUP BY predicted_class;"
            ).fetchall()
            class_dist = [
                {"class_name": r["predicted_class"], "count": r["cnt"], "percentage": round((r["cnt"] / total_count) * 100, 2)}
                for r in class_rows
            ]

            # Averages
            avg_row = conn.execute(
                "SELECT AVG(confidence) as avg_conf, AVG(review_priority) as avg_pri, AVG(inference_latency_ms) as avg_lat FROM predictions;"
            ).fetchone()

            return {
                "total_predictions": total_count,
                "status_distribution": status_dist,
                "priority_distribution": priority_dist,
                "class_distribution": class_dist,
                "average_confidence": round(avg_row["avg_conf"] or 0.0, 4),
                "average_review_priority": round(avg_row["avg_pri"] or 0.0, 4),
                "average_inference_latency_ms": round(avg_row["avg_lat"] or 0.0, 2)
            }

    def _row_to_record(self, row: sqlite3.Row) -> PredictionRecord:
        # Gracefully handle optional legacy columns if any
        keys = row.keys()
        return PredictionRecord(
            id=row["id"],
            prediction_id=row["prediction_id"],
            tile_hash=row["tile_hash"],
            filename=row["filename"],
            predicted_class=row["predicted_class"],
            confidence=row["confidence"],
            top_2_class=row["top_2_class"] if "top_2_class" in keys else "",
            top_2_confidence=row["top_2_confidence"] if "top_2_confidence" in keys else 0.0,
            confidence_margin=row["confidence_margin"] if "confidence_margin" in keys else 0.0,
            review_priority=row["review_priority"] if "review_priority" in keys else 0.0,
            review_priority_level=row["review_priority_level"] if "review_priority_level" in keys else "LOW",
            review_reason=row["review_reason"] if "review_reason" in keys else "STANDARD_CONFIDENCE",
            status=row["status"],
            all_probabilities=row["all_probabilities"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            model_checksum=row["model_checksum"],
            preprocessing_version=row["preprocessing_version"],
            inference_latency_ms=row["inference_latency_ms"],
            total_latency_ms=row["total_latency_ms"],
            created_at=row["created_at"]
        )

prediction_repository = PredictionRepository()

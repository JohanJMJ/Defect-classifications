import csv
import io
import os
from pathlib import Path
import sqlite3
from datetime import datetime

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "inspections.db"

class QCInspectionDB:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_category TEXT NOT NULL,
                    defect_detected INTEGER NOT NULL,
                    defect_class TEXT,
                    confidence REAL,
                    mask_area_pct REAL,
                    verdict TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    rule_action TEXT,
                    llama_advisory TEXT,
                    latency_vision_ms REAL,
                    latency_llama_ms REAL,
                    image_path TEXT,
                    annotated_path TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON inspections(timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_verdict ON inspections(verdict)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON inspections(model_category)")
            conn.commit()

    def add_inspection(self, data):
        # Inserts an inspection record and returns the new inspection ID.
        query = """
            INSERT INTO inspections (
                component_id, timestamp, model_category, defect_detected, defect_class,
                confidence, mask_area_pct, verdict, severity, rule_action, llama_advisory,
                latency_vision_ms, latency_llama_ms, image_path, annotated_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                data.get("component_id", f"PART-{int(datetime.now().timestamp()*1000) % 1000000:06d}"),
                data.get("timestamp", now_str),
                data.get("model_category", "unknown"),
                1 if data.get("defect_detected", False) else 0,
                data.get("defect_class", "None"),
                float(data.get("confidence", 0.0)),
                float(data.get("mask_area_pct", 0.0)),
                data.get("verdict", "PASS"),
                data.get("severity", "NONE"),
                data.get("rule_action", ""),
                data.get("llama_advisory", ""),
                float(data.get("latency_vision_ms", 0.0)),
                float(data.get("latency_llama_ms", 0.0)),
                data.get("image_path", ""),
                data.get("annotated_path", "")
            ))
            conn.commit()
            return cursor.lastrowid

    def get_inspections(self, limit=50, offset=0, verdict=None, model=None, search=None):
        where_clauses = []
        params = []

        if verdict and verdict.upper() != "ALL":
            where_clauses.append("verdict = ?")
            params.append(verdict.upper())

        if model and model.lower() != "all":
            where_clauses.append("model_category = ?")
            params.append(model.lower())

        if search:
            where_clauses.append("(component_id LIKE ? OR defect_class LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        query = f"SELECT * FROM inspections {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = [dict(r) for r in cursor.fetchall()]

            # Also get total count
            count_query = f"SELECT COUNT(*) FROM inspections {where_sql}"
            cursor.execute(count_query, params[:-2])
            total_count = cursor.fetchone()[0]

            return rows, total_count

    def get_inspection_by_id(self, inspection_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_summary_stats(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inspections")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE verdict = 'PASS'")
            passes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE verdict = 'REJECT'")
            rejects = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE verdict = 'FLAGGED_MONITOR'")
            monitors = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(latency_vision_ms) FROM inspections WHERE latency_vision_ms > 0")
            avg_vision_lat = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT AVG(latency_llama_ms) FROM inspections WHERE latency_llama_ms > 0")
            avg_llama_lat = cursor.fetchone()[0] or 0.0

            yield_rate = (passes / total * 100.0) if total > 0 else 100.0

            return {
                "total_inspections": total,
                "pass_count": passes,
                "reject_count": rejects,
                "monitor_count": monitors,
                "yield_rate_pct": round(yield_rate, 1),
                "avg_vision_latency_ms": round(avg_vision_lat, 2),
                "avg_llama_latency_ms": round(avg_llama_lat, 2)
            }

    def export_csv_string(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections ORDER BY id DESC")
            rows = cursor.fetchall()

            output = io.StringIO()
            writer = csv.writer(output)
            if rows:
                writer.writerow(rows[0].keys())
                for r in rows:
                    writer.writerow([r[k] for k in r.keys()])
            else:
                writer.writerow([
                    "id", "component_id", "timestamp", "model_category", "defect_detected",
                    "defect_class", "confidence", "mask_area_pct", "verdict", "severity",
                    "rule_action", "llama_advisory", "latency_vision_ms", "latency_llama_ms",
                    "image_path", "annotated_path"
                ])
            return output.getvalue()

    def update_advisory(self, inspection_id, llama_advisory, latency_llama_ms, severity=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if severity:
                cursor.execute(
                    "UPDATE inspections SET llama_advisory = ?, latency_llama_ms = ?, severity = ? WHERE id = ?",
                    (llama_advisory, float(latency_llama_ms), severity, int(inspection_id))
                )
            else:
                cursor.execute(
                    "UPDATE inspections SET llama_advisory = ?, latency_llama_ms = ? WHERE id = ?",
                    (llama_advisory, float(latency_llama_ms), int(inspection_id))
                )
            conn.commit()

"""
SQLAlchemy database setup and DetectionSession model.
Each time a user runs an analysis, one row is inserted.
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

_DB_PATH = Path(__file__).parent.parent / "safety_log.db"
_DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(
    _DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    image_filename = Column(String(255), nullable=False)
    violation_count = Column(Integer, default=0, nullable=False)
    total_detections = Column(Integer, default=0, nullable=False)
    violations_json = Column(Text, default="[]", nullable=False)   # JSON list

    def set_violations(self, violations: list[dict]) -> None:
        self.violations_json = json.dumps(violations)
        self.violation_count = len(violations)

    def get_violations(self) -> list[dict]:
        return json.loads(self.violations_json)


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def log_session(
    image_filename: str,
    all_detections: list[dict],
) -> DetectionSession:
    """Insert a new detection session row and return it."""
    init_db()
    violations = [d for d in all_detections if d.get("is_violation")]

    session_row = DetectionSession(
        image_filename=image_filename,
        total_detections=len(all_detections),
        violation_count=len(violations),
    )
    session_row.set_violations(violations)

    with SessionLocal() as db:
        db.add(session_row)
        db.commit()
        db.refresh(session_row)
        return session_row


def get_all_sessions() -> list[dict]:
    """Fetch all sessions ordered by timestamp, returned as plain dicts."""
    init_db()
    with SessionLocal() as db:
        rows = db.query(DetectionSession).order_by(DetectionSession.timestamp).all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "image_filename": r.image_filename,
                "violation_count": r.violation_count,
                "total_detections": r.total_detections,
            }
            for r in rows
        ]

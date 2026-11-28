"""Database layer: engine, session factory and the three tables.

DATABASE_URL decides where data lives:
  - Docker / real use: postgresql+psycopg://postgres:<password>@postgres:5432/postgres
  - not set (tests, quick local runs): in-memory SQLite, so no Postgres is needed
"""
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite://")

if DATABASE_URL.startswith("sqlite"):
    # one shared connection, so the in-memory database survives across threads
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(engine, expire_on_commit=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Case(Base):
    """One doctor-patient visit."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Task(Base):
    """One processing run of a case, with the state of each stage."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    status: Mapped[str] = mapped_column(String(16))
    stages: Mapped[dict] = mapped_column(JSON)     # {"asr": "done", "extract": "running", ...}
    errors: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Artifact(Base):
    """The JSON output of one stage for one case (transcript, extraction, ...)."""

    __tablename__ = "artifacts"

    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), primary_key=True)
    stage: Mapped[str] = mapped_column(String(16), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)


def init_db() -> None:
    """Create the tables if they do not exist yet (no migrations tool for now)."""
    Base.metadata.create_all(engine)

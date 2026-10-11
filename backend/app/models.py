"""ORM tables. Works on PostgreSQL (production) and SQLite (local fallback)."""
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    country: Mapped[str] = mapped_column(String(4))
    kyc_level: Mapped[int] = mapped_column(Integer)
    email_domain: Mapped[str] = mapped_column(String(40))
    # Ground-truth scenario from the synthetic generator. Never used as a model feature.
    pattern: Mapped[str] = mapped_column(String(20), default="legit")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    amount: Mapped[float] = mapped_column(Float)
    merchant_id: Mapped[str] = mapped_column(String(40))
    merchant_category: Mapped[str] = mapped_column(String(20))
    channel: Mapped[str] = mapped_column(String(8))
    device_id: Mapped[str] = mapped_column(String(40), index=True)
    ip: Mapped[str] = mapped_column(String(40), index=True)
    ip_country: Mapped[str] = mapped_column(String(4))
    card_id: Mapped[str] = mapped_column(String(40), index=True)
    address_id: Mapped[str] = mapped_column(String(40), index=True)
    # Chargeback label (ground truth for synthetic data) and the generator scenario.
    is_fraud: Mapped[bool] = mapped_column(Boolean, default=False)
    pattern: Mapped[str] = mapped_column(String(20), default="legit")
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.id"), unique=True, index=True)
    account_id: Mapped[str] = mapped_column(String(40), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    severity: Mapped[str] = mapped_column(String(10), index=True)
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)
    scores: Mapped[dict] = mapped_column(JSON)  # per-model component scores
    factors: Mapped[list] = mapped_column(JSON)  # SHAP contributing factors
    narrative: Mapped[str] = mapped_column(Text)
    model_version: Mapped[int] = mapped_column(Integer)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id"), index=True)
    transaction_id: Mapped[str] = mapped_column(String(40), index=True)
    account_id: Mapped[str] = mapped_column(String(40), index=True)
    label: Mapped[str] = mapped_column(String(10))  # "fraud" | "legit"
    analyst: Mapped[str] = mapped_column(String(60), default="analyst")
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    metrics: Mapped[dict] = mapped_column(JSON)
    feature_importance: Mapped[list] = mapped_column(JSON)
    n_train: Mapped[int] = mapped_column(Integer)
    n_feedback: Mapped[int] = mapped_column(Integer, default=0)
    gnn_backend: Mapped[str] = mapped_column(String(30))
    notes: Mapped[str] = mapped_column(Text, default="")

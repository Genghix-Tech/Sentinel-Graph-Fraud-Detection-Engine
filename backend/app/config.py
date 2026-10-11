"""Runtime configuration, driven by environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Postgres in docker-compose / production; SQLite fallback for zero-setup local runs.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'sentinelgraph.db'}")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(BASE_DIR / "artifacts")))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

SEED = int(os.getenv("SEED", "7"))

# Fraud labels (chargebacks) only become known this many days after the transaction.
# Relational "known fraud neighbour" features respect this delay to avoid leakage.
LABEL_DELAY_DAYS = int(os.getenv("LABEL_DELAY_DAYS", "10"))

# Share of transactions the analyst team can review (sets the alert threshold).
ALERT_BUDGET = float(os.getenv("ALERT_BUDGET", "0.03"))

# Alerts are raised for historical transactions inside this recent window at seed time.
ALERT_WINDOW_DAYS = int(os.getenv("ALERT_WINDOW_DAYS", "21"))

# Entities used to link accounts into clusters (IPs are excluded: NAT/office IPs are noisy).
STRONG_ENTITY_TYPES = ("device", "card", "address")

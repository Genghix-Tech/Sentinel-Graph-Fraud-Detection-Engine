from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeedbackIn(BaseModel):
    label: Literal["fraud", "legit"]
    notes: str = ""
    analyst: str = Field("analyst", max_length=60)


class SimulateIn(BaseModel):
    scenario: Literal["legit", "ring", "ato", "card_testing"]


class StreamIn(BaseModel):
    enabled: bool


class TransactionIn(BaseModel):
    """Real-time scoring request. Unknown accounts are created on the fly."""

    account_id: str
    amount: float = Field(gt=0)
    merchant_category: Literal["grocery", "restaurants", "fashion", "travel", "electronics", "gift_cards",
                               "digital_goods", "crypto"]
    channel: Literal["web", "app"] = "web"
    device_id: str
    ip: str
    ip_country: str = Field(min_length=2, max_length=4)
    card_id: str
    address_id: str
    merchant_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    id: Optional[str] = None
    # only used when the account doesn't exist yet
    account_country: str = "US"
    account_kyc_level: int = Field(1, ge=1, le=3)

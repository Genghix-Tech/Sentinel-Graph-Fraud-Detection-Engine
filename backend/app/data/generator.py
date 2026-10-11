"""Synthetic payments world with hidden fraud patterns.

Legitimate customers are deliberately messy (shared family devices, office NAT IPs,
travel, new phones, gifts shipped elsewhere) so that no single signal is a giveaway.
Injected fraud scenarios:

* ``ring``          - synthetic-identity rings: young accounts that share devices, drop
                      addresses and stolen cards, and make *ordinary-looking* purchases.
                      Hard for per-transaction models, obvious in the graph.
* ``ato``           - account takeover: an established account suddenly transacts from a
                      new device in another country, at night, for unusually high amounts.
* ``card_testing``  - bursts of micro-transactions with many cards, then a large cash-out.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CATEGORIES: Dict[str, tuple] = {
    # category: (median amount, lognormal sigma)
    "grocery": (38.0, 0.55),
    "restaurants": (28.0, 0.6),
    "fashion": (70.0, 0.7),
    "travel": (320.0, 0.8),
    "electronics": (180.0, 0.9),
    "gift_cards": (60.0, 0.7),
    "digital_goods": (15.0, 0.9),
    "crypto": (150.0, 1.0),
}
CATEGORY_NAMES = list(CATEGORIES)
LEGIT_CATEGORY_PRIOR = np.array([0.30, 0.22, 0.15, 0.04, 0.08, 0.05, 0.14, 0.02])

HOME_COUNTRIES = ["US", "GB", "CA", "DE", "FR"]
HOME_COUNTRY_P = [0.55, 0.15, 0.12, 0.10, 0.08]
ALL_COUNTRIES = HOME_COUNTRIES + ["ES", "IT", "NL", "JP", "AU", "BR", "MX", "IN", "TR", "ZA", "SG", "PL"]
EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "icloud.com", "proton.me"]

_HOUR_W = np.array([1, .6, .4, .3, .3, .5, 1, 2, 3, 4, 5, 5, 6, 6, 5, 5, 5, 6, 7, 7, 6, 5, 3, 2], dtype=float)
HOUR_P = _HOUR_W / _HOUR_W.sum()
_NIGHT_W = np.array([6, 6, 5, 5, 4, 3, 1, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 1, 1, 2, 3, 4, 5], dtype=float)
NIGHT_P = _NIGHT_W / _NIGHT_W.sum()


@dataclass
class Profile:
    """A legitimate customer and the entities they normally use."""

    account_id: str
    country: str
    created_at: datetime
    devices: List[str]
    home_ip: str
    address: str
    cards: List[str]
    spend_factor: float
    category_pref: np.ndarray
    rate_per_day: float
    mobile_ip: Optional[str] = None
    nat_ip: Optional[str] = None
    shared_device: Optional[str] = None


@dataclass
class World:
    accounts: List[dict] = field(default_factory=list)
    transactions: List[dict] = field(default_factory=list)


class SyntheticWorld:
    def __init__(self, seed: int, end: datetime, id_tag: str = ""):
        self.rng = np.random.default_rng(seed)
        self.end = end
        self.id_tag = id_tag
        self._counters: Dict[str, int] = defaultdict(int)
        self._ips: set = set()
        self.world = World()
        self.ring_devices: List[str] = []

    # ------------------------------------------------------------------ ids
    def new_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}_{self.id_tag}{self._counters[prefix]:05d}"

    def new_ip(self) -> str:
        while True:
            octets = self.rng.integers(1, 255, size=4)
            ip = f"{int(octets[0]) % 222 + 1}.{octets[1]}.{octets[2]}.{octets[3]}"
            if ip not in self._ips:
                self._ips.add(ip)
                return ip

    # ------------------------------------------------------------ primitives
    def _hour_time(self, day: datetime, night: bool = False) -> datetime:
        hour = int(self.rng.choice(24, p=NIGHT_P if night else HOUR_P))
        return day.replace(hour=hour, minute=0, second=0, microsecond=0) + timedelta(
            seconds=float(self.rng.uniform(0, 3600))
        )

    def _amount(self, category: str, factor: float = 1.0) -> float:
        median, sigma = CATEGORIES[category]
        return round(float(self.rng.lognormal(np.log(median * factor), sigma)), 2)

    def add_account(self, created_at: datetime, country: str, kyc: int, pattern: str) -> str:
        acc_id = self.new_id("acc")
        self.world.accounts.append(
            {
                "id": acc_id,
                "created_at": created_at,
                "country": country,
                "kyc_level": kyc,
                "email_domain": str(self.rng.choice(EMAIL_DOMAINS)),
                "pattern": pattern,
            }
        )
        return acc_id

    def add_txn(self, account_id: str, ts: datetime, amount: float, category: str, device: str, ip: str,
                ip_country: str, card: str, address: str, is_fraud: bool, pattern: str,
                channel: Optional[str] = None) -> dict:
        txn = {
            "id": self.new_id("txn"),
            "account_id": account_id,
            "timestamp": ts,
            "amount": max(0.5, round(amount, 2)),
            "merchant_id": f"m_{category}_{int(self.rng.integers(0, 20)):02d}",
            "merchant_category": category,
            "channel": channel or str(self.rng.choice(["web", "app"], p=[0.45, 0.55])),
            "device_id": device,
            "ip": ip,
            "ip_country": ip_country,
            "card_id": card,
            "address_id": address,
            "is_fraud": is_fraud,
            "pattern": pattern,
        }
        self.world.transactions.append(txn)
        return txn

    # -------------------------------------------------------- legit traffic
    def make_profile(self, created_at: datetime) -> Profile:
        country = str(self.rng.choice(HOME_COUNTRIES, p=HOME_COUNTRY_P))
        new_customer = (self.end - created_at).days < 60
        kyc = int(self.rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15] if new_customer else [0.1, 0.45, 0.45]))
        acc = self.add_account(created_at, country, kyc, "legit")
        return Profile(
            account_id=acc,
            country=country,
            created_at=created_at,
            devices=[self.new_id("dev") for _ in range(1 + int(self.rng.random() < 0.35))],
            home_ip=self.new_ip(),
            mobile_ip=self.new_ip() if self.rng.random() < 0.5 else None,
            address=self.new_id("addr"),
            cards=[self.new_id("card") for _ in range(1 + int(self.rng.random() < 0.3))],
            spend_factor=float(self.rng.lognormal(0, 0.35)),
            category_pref=self.rng.dirichlet(LEGIT_CATEGORY_PRIOR * 20),
            rate_per_day=float(np.clip(self.rng.lognormal(np.log(0.12), 0.7), 0.01, 1.5)),
        )

    def legit_txn(self, p: Profile, ts: datetime) -> dict:
        rng = self.rng
        category = str(rng.choice(CATEGORY_NAMES, p=p.category_pref))
        device = p.devices[int(rng.integers(0, len(p.devices)))]
        if p.shared_device and rng.random() < 0.25:
            device = p.shared_device
        ip, ip_country = p.home_ip, p.country
        r = rng.random()
        if p.nat_ip and r < 0.25:
            ip = p.nat_ip
        elif p.mobile_ip and r < 0.45:
            ip = p.mobile_ip
        address = p.address
        if rng.random() < 0.02:  # travelling: hotel wifi abroad
            ip = self.new_ip()
            ip_country = str(rng.choice([c for c in ALL_COUNTRIES if c != p.country]))
        if rng.random() < 0.012:  # bought a new phone
            device = self.new_id("dev")
            p.devices.append(device)
        if rng.random() < 0.02:  # gift shipped to someone else
            address = self.new_id("addr")
        return self.add_txn(p.account_id, ts, self._amount(category, p.spend_factor), category, device, ip,
                            ip_country, p.cards[int(rng.integers(0, len(p.cards)))], address, False, "legit")

    def legit_history(self, p: Profile, start: datetime) -> None:
        t0 = max(start, p.created_at)
        active_days = (self.end - t0).total_seconds() / 86400
        if active_days <= 0:
            return
        n = int(self.rng.poisson(p.rate_per_day * active_days))
        for offset in np.sort(self.rng.uniform(0, active_days, n)):
            day = t0 + timedelta(days=float(offset))
            ts = self._hour_time(day)
            if ts > p.created_at and ts < self.end:
                self.legit_txn(p, ts)

    def link_households(self, profiles: Sequence[Profile]) -> None:
        """Families share a tablet, home wifi and address - benign entity sharing."""
        idx = self.rng.permutation(len(profiles))[: int(len(profiles) * 0.12)]
        i = 0
        while i < len(idx) - 1:
            size = int(self.rng.integers(2, 5))
            members = [profiles[j] for j in idx[i:i + size]]
            device = self.new_id("dev")
            for m in members:
                m.shared_device = device
                m.home_ip = members[0].home_ip
                m.address = members[0].address
                m.country = members[0].country
            i += size

    def student_housing(self, profiles: Sequence[Profile], start: datetime, days: int, n_groups: int = 18) -> None:
        """Hard negatives: roommates who open accounts together and share an address and wifi.

        Structurally this looks like a young fraud ring (new accounts, created within days of each
        other, shared address/IP) but they use their own phones and cards and spend normally."""
        for _ in range(n_groups):
            move_in = start + timedelta(days=float(self.rng.uniform(0, days - 10)))
            address, ip = self.new_id("addr"), self.new_ip()
            for _ in range(int(self.rng.integers(3, 7))):
                p = self.make_profile(move_in + timedelta(days=float(self.rng.uniform(0, 6))))
                p.address, p.home_ip = address, ip
                p.rate_per_day *= 1.5
                profiles.append(p)

    def second_hand_devices(self, profiles: Sequence[Profile], share: float = 0.03) -> None:
        """Resold phones: a device that belonged to one customer later shows up on another."""
        idx = self.rng.permutation(len(profiles))
        n = int(len(profiles) * share)
        for seller, buyer in zip(idx[:n], idx[n:2 * n]):
            profiles[buyer].devices.append(profiles[seller].devices[0])

    def assign_nat_ips(self, profiles: Sequence[Profile], n_nat: int = 8) -> None:
        """Offices/campuses: many unrelated customers behind one public IP."""
        for _ in range(n_nat):
            ip = self.new_ip()
            for j in self.rng.choice(len(profiles), size=int(self.rng.integers(15, 41)), replace=False):
                profiles[j].nat_ip = ip

    # -------------------------------------------------------- fraud scenarios
    def fraud_ring(self, t_start: datetime, size: Optional[int] = None,
                   reuse_devices: Sequence[str] = ()) -> List[str]:
        rng = self.rng
        size = size or int(rng.integers(4, 11))
        country = str(rng.choice(HOME_COUNTRIES, p=HOME_COUNTRY_P))
        devices = [self.new_id("dev") for _ in range(int(rng.integers(2, 4)))]
        if reuse_devices:
            devices[0] = str(rng.choice(list(reuse_devices)))
        elif self.ring_devices and rng.random() < 0.35:  # fraudsters recycle hardware between rings
            devices[0] = str(rng.choice(self.ring_devices))
        self.ring_devices.extend(devices)
        ips = [self.new_ip() for _ in range(int(rng.integers(2, 5)))]
        drops = [self.new_id("addr") for _ in range(int(rng.integers(1, 3)))]
        card_pool = [self.new_id("card") for _ in range(size + int(rng.integers(0, 3)))]
        cash_cats = ["electronics", "gift_cards", "fashion", "digital_goods", "grocery", "restaurants"]
        cash_p = [.25, .15, .25, .15, .1, .1]

        members = []
        for _ in range(size):
            # half are fresh synthetic identities, half are aged "sleeper" accounts bought or farmed earlier
            age = rng.uniform(1, 21) if rng.random() < 0.5 else rng.uniform(25, 160)
            created = t_start - timedelta(days=float(age))
            acc = self.add_account(created, country, int(rng.choice([1, 2, 3], p=[0.4, 0.45, 0.15])), "ring")
            members.append(acc)
            my_devices = list(rng.choice(devices, size=min(2, len(devices)), replace=False))
            my_cards = list(rng.choice(card_pool, size=int(rng.integers(1, 3)), replace=False))

            def emit(ts: datetime, category: str, amount: float) -> None:
                if ts < self.end:
                    self.add_txn(acc, ts, amount, category, str(rng.choice(my_devices)), str(rng.choice(ips)),
                                 country, str(rng.choice(my_cards)), str(rng.choice(drops)), True, "ring")

            act_start = t_start + timedelta(days=float(rng.uniform(0, 4)))
            for _ in range(int(rng.integers(1, 4))):  # warm-up purchases to look normal
                dormant_until = max(created, act_start - timedelta(days=30))  # sleepers wake up shortly before
                day = dormant_until + timedelta(days=float(rng.uniform(0.3, max(0.4, (act_start - dormant_until).days))))
                cat = str(rng.choice(["grocery", "restaurants", "digital_goods"]))
                emit(self._hour_time(day), cat, self._amount(cat, 0.6))
            span = float(rng.uniform(1, 5))
            for _ in range(int(rng.integers(3, 9))):  # cash-out with ordinary-looking amounts
                day = act_start + timedelta(days=float(rng.uniform(0, span)))
                cat = str(rng.choice(cash_cats, p=cash_p))
                amount = float(np.clip(rng.lognormal(np.log(85), 0.7), 8, 1500))
                emit(self._hour_time(day), cat, amount)
        return members

    def account_takeover(self, p: Profile, t: datetime, mule_addresses: Sequence[str]) -> None:
        rng = self.rng
        device, ip = self.new_id("dev"), self.new_ip()
        ip_country = str(rng.choice([c for c in ALL_COUNTRIES if c != p.country]))
        address = str(rng.choice(mule_addresses)) if rng.random() < 0.5 else self.new_id("addr")
        t = self._hour_time(t, night=rng.random() < 0.6)
        for offset in np.sort(rng.uniform(0, 6 * 3600, int(rng.integers(2, 6)))):
            ts = t + timedelta(seconds=float(offset))
            if ts >= self.end:
                break
            cat = str(rng.choice(["electronics", "gift_cards", "crypto"], p=[.4, .3, .3]))
            amount = float(np.clip(rng.lognormal(np.log(60 * p.spend_factor * 6), 0.5), 250, 5000))
            self.add_txn(p.account_id, ts, amount, cat, device, ip, ip_country, p.cards[0], address, True, "ato")

    def card_testing(self, t: datetime, devices: Sequence[str], proxy_ips: Sequence[str]) -> str:
        rng = self.rng
        country = str(rng.choice(HOME_COUNTRIES, p=HOME_COUNTRY_P))
        acc = self.add_account(t - timedelta(days=float(rng.uniform(0.01, 2))), country, 1, "card_testing")
        device, ip = str(rng.choice(devices)), str(rng.choice(proxy_ips))
        ip_country = str(rng.choice(ALL_COUNTRIES))
        address = self.new_id("addr")
        ts, card = t, None
        for _ in range(int(rng.integers(12, 30))):
            ts = ts + timedelta(seconds=float(rng.uniform(10, 70)))
            card = self.new_id("card")
            if ts < self.end:
                self.add_txn(acc, ts, float(rng.uniform(0.5, 4.99)), "digital_goods", device, ip, ip_country,
                             card, address, True, "card_testing", channel="web")
        for _ in range(int(rng.integers(1, 3))):
            ts = ts + timedelta(minutes=float(rng.uniform(5, 40)))
            if ts < self.end:
                self.add_txn(acc, ts, float(rng.uniform(200, 900)), "electronics", device, ip, ip_country,
                             card, address, True, "card_testing", channel="web")
        return acc

    # ------------------------------------------------------------ full world
    def generate(self, n_accounts: int = 2500, days: int = 90, n_rings: int = 16, n_ato: int = 30,
                 n_card_testing: int = 8) -> Dict[str, pd.DataFrame]:
        rng = self.rng
        start = self.end - timedelta(days=days)
        profiles = []
        for _ in range(n_accounts):
            if rng.random() < 0.20:  # new customers who joined during the window
                created = start + timedelta(days=float(rng.uniform(0, days - 3)))
            else:
                created = start - timedelta(days=float(rng.uniform(5, 700)))
            profiles.append(self.make_profile(created))
        self.link_households(profiles)
        self.student_housing(profiles, start, days)
        self.second_hand_devices(profiles)
        self.assign_nat_ips(profiles)
        for p in profiles:
            self.legit_history(p, start)

        for t in np.linspace(3, days - 6, n_rings) + rng.uniform(-2, 2, n_rings):
            self.fraud_ring(start + timedelta(days=float(t)))

        mules = [self.new_id("addr") for _ in range(3)]
        established = [p for p in profiles if (start - p.created_at).days > 60]
        for j in rng.choice(len(established), size=n_ato, replace=False):
            self.account_takeover(established[j], start + timedelta(days=float(rng.uniform(2, days - 1))), mules)

        ct_devices = [self.new_id("dev") for _ in range(3)]
        proxies = [self.new_ip() for _ in range(4)]
        for t in rng.uniform(2, days - 1, n_card_testing):
            self.card_testing(start + timedelta(days=float(t)), ct_devices, proxies)

        return self.frames()

    def frames(self) -> Dict[str, pd.DataFrame]:
        accounts = pd.DataFrame(self.world.accounts)
        txns = pd.DataFrame(self.world.transactions).sort_values("timestamp").reset_index(drop=True)
        return {"accounts": accounts, "transactions": txns}


def generate_world(seed: int, end: Optional[datetime] = None, **kwargs) -> Dict[str, pd.DataFrame]:
    # anchored to midnight UTC so a given seed always produces the same world
    end = end or datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return SyntheticWorld(seed, end).generate(**kwargs)

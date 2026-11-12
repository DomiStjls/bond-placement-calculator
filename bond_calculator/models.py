"""Параметры сделки и системы. Деньги — рубли, ставки — доли единицы."""

from dataclasses import dataclass, fields
from datetime import date
import json
from math import isfinite
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name}: требуется конечное неотрицательное число")


def month_date(value: date) -> None:
    if type(value) is not date or value.day != 1:
        raise ValueError("Месяц должен быть датой первого числа")


def add_months(value: date, offset: int) -> date:
    total = value.year * 12 + value.month - 1 + offset
    return date(total // 12, total % 12 + 1, 1)


@dataclass(frozen=True)
class Deal:
    id: str
    name: str
    rating: str
    issue_volume: float
    term_months: int
    placement_month: date
    coupon_type: str
    coupon_spread: float | None
    fixed_coupon_rate: float | None
    organizer_fee_rate: float
    annual_pd: float
    borrower_risk_weight: float
    free_limit: float
    institutional_demand: float
    retail_demand: float
    monthly_resale_volume: float

    def __post_init__(self):
        month_date(self.placement_month)
        if type(self.term_months) is not int or self.term_months < 1:
            raise ValueError("term_months: требуется положительное целое число")
        for name in ("issue_volume", "organizer_fee_rate", "annual_pd", "borrower_risk_weight",
                     "free_limit", "institutional_demand", "retail_demand", "monthly_resale_volume"):
            nonnegative(name, getattr(self, name))
        if self.annual_pd > 1 or self.organizer_fee_rate > 1:
            raise ValueError("PD и комиссия задаются долями единицы, не процентами")
        if self.coupon_type == "floating":
            nonnegative("coupon_spread", self.coupon_spread)
            if self.fixed_coupon_rate is not None:
                raise ValueError("Для плавающего купона fixed_coupon_rate должен быть null")
        elif self.coupon_type == "fixed":
            nonnegative("fixed_coupon_rate", self.fixed_coupon_rate)
            if self.coupon_spread is not None:
                raise ValueError("Для фиксированного купона coupon_spread должен быть null")
        else:
            raise ValueError("coupon_type: требуется fixed или floating")


@dataclass(frozen=True)
class RatePeriod:
    effective_from: date
    annual_rate: float

    def __post_init__(self):
        month_date(self.effective_from)
        nonnegative("annual_rate", self.annual_rate)


@dataclass(frozen=True)
class SystemParameters:
    key_rates: tuple[RatePeriod, ...]
    tax_rate: float = 0.25
    funding_spread: float = 0.002
    loss_given_default: float = 0.45
    capital_ratio: float = 0.12
    resale_risk_weight: float = 1.2
    bank_placement_fee_share: float = 0.5
    institutional_rebate_share: float = 0.9
    retail_ecosystem_rate: float = 0.01
    resale_income_rate: float = 0.0054
    own_balance_placement_fee_rate: float = 0.0
    resale_start_month: int = 2
    allocation_step: int = 1_000_000
    irr_threshold: float = 0.30
    roe_threshold: float = 0.265

    def __post_init__(self):
        for field in fields(self):
            if field.name != "key_rates":
                nonnegative(field.name, getattr(self, field.name))
        for name in ("tax_rate", "loss_given_default", "capital_ratio", "bank_placement_fee_share",
                     "institutional_rebate_share", "retail_ecosystem_rate", "resale_income_rate",
                     "own_balance_placement_fee_rate"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name}: требуется доля от 0 до 1")
        if type(self.resale_start_month) is not int or self.resale_start_month < 2:
            raise ValueError("Перепродажа допускается не ранее второго месяца")
        if type(self.allocation_step) is not int or self.allocation_step <= 0:
            raise ValueError("allocation_step: требуется положительное целое число рублей")
        if not isinstance(self.key_rates, tuple) or not self.key_rates:
            raise ValueError("key_rates: требуется непустой tuple периодов")
        if any(not isinstance(p, RatePeriod) for p in self.key_rates):
            raise ValueError("key_rates должен содержать RatePeriod")
        dates = [p.effective_from for p in self.key_rates]
        if dates != sorted(set(dates)):
            raise ValueError("Периоды КС должны идти по возрастанию без повторов")

    def key_rate_at(self, month: date) -> float:
        month_date(month)
        for period in reversed(self.key_rates):
            if period.effective_from <= month:
                return period.annual_rate
        raise ValueError(f"Нет ключевой ставки на {month.isoformat()}")


def load_deals(path: str | Path | None = None) -> dict[str, Deal]:
    payload = json.loads(Path(path or DATA_DIR / "deals.json").read_text(encoding="utf-8"))
    if (payload["schema_version"], payload["money_unit"], payload["rate_format"]) != (1, "RUB", "decimal"):
        raise ValueError("Неподдерживаемый формат данных сделок")
    deals = {}
    for row in payload["deals"]:
        row = dict(row)
        row["placement_month"] = date.fromisoformat(row["placement_month"])
        deal = Deal(**row)
        if deal.id in deals:
            raise ValueError(f"Повтор id сделки: {deal.id}")
        deals[deal.id] = deal
    return deals


def load_system(path: str | Path | None = None) -> SystemParameters:
    payload = json.loads(Path(path or DATA_DIR / "system.json").read_text(encoding="utf-8"))
    if (payload["schema_version"], payload["rate_format"]) != (1, "decimal"):
        raise ValueError("Неподдерживаемый формат системных параметров")
    params = dict(payload["system"])
    params["key_rates"] = tuple(
        RatePeriod(date.fromisoformat(row["effective_from"]), row["annual_rate"])
        for row in params["key_rates"]
    )
    return SystemParameters(**params)

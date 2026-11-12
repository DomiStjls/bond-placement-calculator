"""Четыре канала; x передаётся отдельно от Deal и SystemParameters.

Налог в ChannelResult относится к каналу в изоляции. Для сделки использовать
calculate_allocation: сначала суммируются месячные прибыли ДО налога.
Возврат номинала и капитала не входит в прибыль.
"""

from dataclasses import dataclass
from datetime import date
from math import fsum

from .models import Deal, SystemParameters, add_months, nonnegative


@dataclass(frozen=True)
class MonthlyResult:
    month: date
    opening_position: float = 0.0
    sold_principal: float = 0.0
    redeemed_principal: float = 0.0
    closing_position: float = 0.0
    opening_capital: float = 0.0
    closing_capital: float = 0.0
    placement_fee: float = 0.0
    institutional_rebate: float = 0.0
    ecosystem_income: float = 0.0
    resale_income: float = 0.0
    coupon_income: float = 0.0
    funding_cost: float = 0.0
    expected_loss: float = 0.0
    capital_income: float = 0.0
    profit_before_tax: float = 0.0
    tax: float = 0.0
    net_profit: float = 0.0


@dataclass(frozen=True)
class ChannelResult:
    channel: str
    x: float
    months: tuple[MonthlyResult, ...]

    @property
    def profit_before_tax(self) -> float:
        return fsum(m.profit_before_tax for m in self.months)

    @property
    def standalone_net_profit(self) -> float:
        """Прибыль после налога, если оценивать только этот канал."""
        return fsum(m.net_profit for m in self.months)


@dataclass(frozen=True)
class AllocationResult:
    channels: tuple[ChannelResult, ...]
    months: tuple[MonthlyResult, ...]

    @property
    def profit_before_tax(self) -> float:
        return fsum(m.profit_before_tax for m in self.months)

    @property
    def net_profit(self) -> float:
        return fsum(m.net_profit for m in self.months)


def _validate_x(x: float, deal: Deal, cap: float) -> None:
    nonnegative("x", x)
    if x > min(deal.issue_volume, cap):
        raise ValueError("x превышает объём выпуска, спрос или лимит канала")


def _monthly(month: date, system: SystemParameters, **items) -> MonthlyResult:
    profit = fsum(items.get(k, 0.0) for k in (
        "placement_fee", "ecosystem_income", "resale_income", "coupon_income", "capital_income"
    )) - fsum(items.get(k, 0.0) for k in ("institutional_rebate", "funding_cost", "expected_loss"))
    tax = max(profit, 0.0) * system.tax_rate
    return MonthlyResult(month, **items, profit_before_tax=profit, tax=tax, net_profit=profit-tax)


def institutional_placement(x: float, deal: Deal, system: SystemParameters) -> ChannelResult:
    """x1: комиссия банка минус возврат институционалам; доход в первом месяце."""
    _validate_x(x, deal, deal.institutional_demand)
    fee = x * deal.organizer_fee_rate * system.bank_placement_fee_share
    months = tuple(
        _monthly(add_months(deal.placement_month, i), system,
                 placement_fee=fee if i == 0 else 0.0,
                 institutional_rebate=fee * system.institutional_rebate_share if i == 0 else 0.0)
        for i in range(deal.term_months)
    )
    return ChannelResult("institutional_placement", x, months)


def retail_placement(x: float, deal: Deal, system: SystemParameters) -> ChannelResult:
    """x2: комиссия банка и 1% экосистемного дохода, без балансовой позиции."""
    _validate_x(x, deal, deal.retail_demand)
    months = tuple(
        _monthly(add_months(deal.placement_month, i), system,
                 placement_fee=x * deal.organizer_fee_rate * system.bank_placement_fee_share if i == 0 else 0.0,
                 ecosystem_income=x * system.retail_ecosystem_rate if i == 0 else 0.0)
        for i in range(deal.term_months)
    )
    return ChannelResult("retail_placement", x, months)


def _balance(x: float, deal: Deal, system: SystemParameters, *, resale: bool) -> ChannelResult:
    _validate_x(x, deal, deal.free_limit)
    weight = system.resale_risk_weight if resale else deal.borrower_risk_weight
    capital_fraction = weight * system.capital_ratio
    if capital_fraction > 1:
        raise ValueError("Требуемый капитал превышает позицию; эта модель такого не поддерживает")
    position = x
    months = []
    for i in range(deal.term_months):
        month = add_months(deal.placement_month, i)
        key_rate = system.key_rate_at(month)
        coupon_rate = key_rate + deal.coupon_spread if deal.coupon_type == "floating" else deal.fixed_coupon_rate
        funding_rate = key_rate + system.funding_spread
        capital = position * capital_fraction
        sold = min(position, deal.monthly_resale_volume) if resale and i + 1 >= system.resale_start_month else 0.0
        redeemed = position - sold if i == deal.term_months - 1 else 0.0
        closing = position - sold - redeemed
        months.append(_monthly(
            month, system, opening_position=position, sold_principal=sold,
            redeemed_principal=redeemed, closing_position=closing,
            opening_capital=capital, closing_capital=closing * capital_fraction,
            placement_fee=x * system.own_balance_placement_fee_rate if i == 0 else 0.0,
            resale_income=sold * system.resale_income_rate,
            coupon_income=position * coupon_rate / 12,
            funding_cost=position * funding_rate / 12,
            expected_loss=position * deal.annual_pd * system.loss_given_default / 12,
            capital_income=capital * funding_rate / 12,
        ))
        position = closing
    return ChannelResult("balance_for_resale" if resale else "hold_to_maturity", x, tuple(months))


def balance_for_resale(x: float, deal: Deal, system: SystemParameters) -> ChannelResult:
    """x3: ежемесячная перепродажа с месяца 2, остаток погашается эмитентом."""
    return _balance(x, deal, system, resale=True)


def hold_to_maturity(x: float, deal: Deal, system: SystemParameters) -> ChannelResult:
    """x4: полная позиция до погашения, риск-вес заёмщика."""
    return _balance(x, deal, system, resale=False)


def calculate_allocation(
    x1: float, x2: float, x3: float, x4: float,
    deal: Deal, system: SystemParameters,
) -> AllocationResult:
    """Рассчитать заданное распределение, без поиска оптимума и проверки IRR/ROE.

    Каналы допускают произвольный неотрицательный x для исследования экономики.
    Здесь распределение дополнительно проверяется на согласованный шаг 1 млн.
    Налог начисляется после суммирования прибыли каналов за каждый месяц.
    """
    for x in (x1, x2, x3, x4):
        nonnegative("x", x)
        if x % system.allocation_step != 0:
            raise ValueError("Объёмы распределения должны быть кратны allocation_step")
    if fsum((x1, x2, x3, x4)) > deal.issue_volume:
        raise ValueError("Сумма каналов превышает объём выпуска")
    if fsum((x3, x4)) > deal.free_limit:
        raise ValueError("Суммарная позиция на балансе превышает свободный лимит")
    channels = (
        institutional_placement(x1, deal, system), retail_placement(x2, deal, system),
        balance_for_resale(x3, deal, system), hold_to_maturity(x4, deal, system),
    )
    component_names = (
        "opening_position", "sold_principal", "redeemed_principal", "closing_position",
        "opening_capital", "closing_capital", "placement_fee", "institutional_rebate",
        "ecosystem_income", "resale_income", "coupon_income", "funding_cost",
        "expected_loss", "capital_income",
    )
    months = tuple(
        _monthly(add_months(deal.placement_month, i), system, **{
            name: fsum(getattr(c.months[i], name) for c in channels)
            for name in component_names
        })
        for i in range(deal.term_months)
    )
    return AllocationResult(channels, months)

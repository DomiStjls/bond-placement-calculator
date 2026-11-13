"""Доходность собственного капитала; ставки в долях единицы."""
from dataclasses import dataclass
from math import exp, expm1, fsum, isfinite, log1p
from .channels import AllocationResult
from .models import SystemParameters

RATE_TOL = 1e-9


@dataclass(frozen=True)
class IRRResult:
    monthly: float | None
    annual: float | None
    status: str


def calculate_irr(flows) -> IRRResult:
    """Уникальная IRR при одной смене знака − → +.

    Отрицательные выплаты до начала положительных поддерживаются.
    Несколько смен знака требуют отдельного анализа; произвольный корень не выбираем.
    """
    flows = tuple(float(v) for v in flows)
    if not flows or not all(isfinite(v) for v in flows):
        raise ValueError("Денежные потоки должны быть конечными и непустыми")
    nonzero = [v for v in flows if v != 0]
    if not nonzero or not any(v < 0 for v in flows) or not any(v > 0 for v in flows):
        return IRRResult(None, None, "no_root")
    changes = sum(a*b < 0 for a,b in zip(nonzero, nonzero[1:]))
    if changes != 1 or nonzero[0] >= 0:
        return IRRResult(None, None, "nonconventional")
    # y = log(1+r_month). Масштабирование экспонент предотвращает переполнение.
    scale = max(abs(v) for v in flows)
    def npv_scaled(y):
        powers = [-i*y for i in range(len(flows))]
        shift = max(p for p,v in zip(powers, flows) if v)
        return fsum((v/scale)*exp(p-shift) for v,p in zip(flows,powers) if v)
    lo, hi = -1., 1.
    for _ in range(20):
        if npv_scaled(lo) > 0 and npv_scaled(hi) < 0:
            break
        lo *= 2; hi *= 2
    else:
        return IRRResult(None, None, "numerical_failure")
    for _ in range(160):
        mid = (lo+hi)/2
        if npv_scaled(mid) > 0: lo = mid
        else: hi = mid
    try:
        monthly, annual = expm1((lo+hi)/2), expm1(12*(lo+hi)/2)
    except OverflowError:
        return IRRResult(None, None, "numerical_failure")
    return IRRResult(monthly, annual, "unique")


@dataclass(frozen=True)
class ReturnMetrics:
    net_profit: float
    capital_years: float
    roe: float | None
    monthly_annualized_roe: tuple[float | None, ...]
    irr: IRRResult
    equity_cash_flows: tuple[float, ...]
    npv_at_irr_threshold: float
    roe_passed: bool | None
    irr_passed: bool | None
    accepted: bool
    status: str
    first_month_profit_share: float | None
    warnings: tuple[str, ...]


def calculate_returns(allocation: AllocationResult, system: SystemParameters) -> ReturnMetrics:
    months = allocation.months
    if not months:
        raise ValueError("Нет месяцев для расчёта доходности")
    profit = allocation.net_profit
    capital_years = fsum(m.opening_capital for m in months)/12
    flows = (-months[0].opening_capital,) + tuple(
        m.net_profit + m.opening_capital - m.closing_capital for m in months)
    discount_log = log1p(system.irr_threshold)/12
    npv = fsum(v*exp(-i*discount_log) for i,v in enumerate(flows))
    monthly_roe = tuple(12*m.net_profit/m.opening_capital if m.opening_capital > 0 else None for m in months)
    warnings = []
    if months[0].placement_fee - months[0].institutional_rebate + months[0].ecosystem_income > 0:
        warnings.append("Доход от размещения поступает один раз, в первом месяце. Он включён в общую прибыль и ROE.")
    share = months[0].net_profit/profit if profit > 0 else None
    if capital_years == 0:
        accepted = profit > 0
        return ReturnMetrics(profit,0,None,monthly_roe,IRRResult(None,None,"no_capital"),
            flows,npv,None,None,accepted,"accepted_no_capital" if accepted else "no_positive_profit",share,tuple(warnings))
    roe = profit/capital_years
    irr = calculate_irr(flows)
    roe_ok = roe + RATE_TOL >= system.roe_threshold
    irr_ok = irr.annual + RATE_TOL >= system.irr_threshold if irr.annual is not None else None
    accepted = profit > 0 and roe_ok and irr_ok is True
    status = "accepted" if accepted else ("irr_requires_review" if irr_ok is None else "thresholds_failed")
    if irr.status == "nonconventional":
        warnings.append("Вложения и поступления чередуются. Для такого денежного потока IRR нужно проверить отдельно.")
    return ReturnMetrics(profit,capital_years,roe,monthly_roe,irr,flows,npv,roe_ok,irr_ok,
                         accepted,status,share,tuple(warnings))

"""MILP по участкам перепродажи, с ROE и NPV-ограничением для IRR.

Кандидат всегда проверяется исходным калькулятором и calculate_returns.
Если оптимум релаксации имеет нетипичные потоки, точный результат не заявляется.
"""
from dataclasses import dataclass
from math import ceil, floor, isfinite
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from .models import Deal, SystemParameters, add_months
from .channels import AllocationResult, calculate_allocation
from .returns import ReturnMetrics, calculate_returns


@dataclass(frozen=True)
class OptimizationResult:
    volumes: tuple[int, int, int, int] | None
    allocation: AllocationResult | None
    metrics: ReturnMetrics | None
    status: str
    optimality_proven: bool
    upper_bound_profit: float | None
    absolute_gap: float | None
    elapsed_seconds: float
    solved_intervals: int
    total_intervals: int
    warnings: tuple[str, ...]


def optimize_milp(deal: Deal, system: SystemParameters, *, time_limit: float = 15.) -> OptimizationResult:
    """Общий лимит времени для MILP-решений, включая все интервалы.

    optimal: максимум среди принимаемых моделей подтверждён в числовых допусках.
    incomplete: найден только проверенный кандидат, доказательство не завершено.
    no_acceptable_allocation: доказано отсутствие положительного допустимого варианта.
    """
    if not isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit должен быть положительным конечным числом")
    started = perf_counter()
    unit = system.allocation_step
    n = deal.term_months
    weights = np.array([system.resale_risk_weight, deal.borrower_risk_weight])*system.capital_ratio
    if np.any(weights > 1):
        raise ValueError("Капитал превышает позицию; модель этого не поддерживает")
    ub = np.floor(np.array([min(deal.issue_volume,deal.institutional_demand),
        min(deal.issue_volume,deal.retail_demand),min(deal.issue_volume,deal.free_limit),
        min(deal.issue_volume,deal.free_limit)])/unit)
    limits = np.array([deal.issue_volume,deal.free_limit])/unit
    q = deal.monthly_resale_volume/unit
    opportunities = np.maximum(np.arange(n)-(system.resale_start_month-1),0)
    selling = np.arange(n)+1 >= system.resale_start_month
    rates = np.array([system.key_rate_at(add_months(deal.placement_month,i)) for i in range(n)])
    funding = rates+system.funding_spread
    coupon = rates+deal.coupon_spread if deal.coupon_type == "floating" else np.full(n,deal.fixed_coupon_rate)
    margin = coupon-funding-deal.annual_pd*system.loss_given_default
    P = np.zeros((n,4)); K = np.zeros((n,4))
    P[0,0] = deal.organizer_fee_rate*system.bank_placement_fee_share*(1-system.institutional_rebate_share)
    P[0,1] = deal.organizer_fee_rate*system.bank_placement_fee_share+system.retail_ecosystem_rate
    P[:,3] = (margin+weights[1]*funding)/12
    P[0,3] += system.own_balance_placement_fee_rate
    K[:,3] = weights[1]
    def resale(x):
        opening = np.maximum(x-q*opportunities,0)
        sold = np.where(selling,np.minimum(opening,q),0)
        profit = opening*(margin+weights[0]*funding)/12 + sold*system.resale_income_rate
        profit[0] += x*system.own_balance_placement_fee_rate
        return profit,opening*weights[0]
    cap = ub[2]
    # Все изменения opening/sold, включая изменяемый первый месяц продаж.
    counts = set(opportunities.tolist()) | set((opportunities+selling).tolist())
    knots = sorted(set([0.,cap]+[q*k for k in counts if 0 < q*k < cap]))
    intervals = [(a,b) for a,b in zip(knots[:-1],knots[1:]) if ceil(a)<=floor(b)]
    if cap == 0: intervals = [(0.,0.)]
    # При нулевом риск-весе/нормативе балансовая позиция тоже может не требовать
    # капитала. Для неё действует только положительная прибыль, без NPV-порога.
    intervals = [(lo,hi,False) for lo,hi in intervals] + (
        [(lo,hi,True) for lo,hi in intervals] if np.any(weights==0) else [])
    best = None; checked = 0; upper_bounds = []; warnings = []
    # Надёжный комиссионный кандидат, в том числе если MILP остановится по времени.
    remaining = floor(limits[0]); direct = [0,0,0,0]
    for j in sorted((0,1),key=lambda j:P[0,j],reverse=True):
        direct[j] = int(min(ub[j],remaining)); remaining -= direct[j]
    def evaluate(x):
        volumes = tuple(int(v)*unit for v in x)
        result = calculate_allocation(*volumes,deal,system)
        return volumes,result,calculate_returns(result,system)
    candidate = evaluate(direct)
    if candidate[2].accepted: best=candidate
    discount = np.exp(-np.arange(1,n+1)*np.log1p(system.irr_threshold)/12)
    # NPV = discount @ profit + capital_coeff @ K_begin.
    capital_coeff = np.r_[discount[0]-1, np.diff(discount)]
    all_solved = True
    for lo,hi,capital_free in intervals:
        remaining_time = time_limit-(perf_counter()-started)
        if remaining_time <= 0:
            all_solved=False; break
        pl,kl = resale(lo)
        if hi > lo:
            ph,kh = resale(hi); ps=(ph-pl)/(hi-lo); ks=(kh-kl)/(hi-lo)
        else:
            ps=np.zeros(n); ks=np.zeros(n)
        p0=pl-ps*lo; k0=kl-ks*lo
        P[:,2]=ps; K[:,2]=ks
        matrices=[np.column_stack([np.array([[1,1,1,1],[0,0,1,1]]),np.zeros((2,n))]),
                  np.column_stack([-P,np.eye(n)]),np.column_stack([-(1-system.tax_rate)*P,np.eye(n)])]
        rhs=[limits,p0,(1-system.tax_rate)*p0]
        # ROE: sum(z) >= threshold * capital-years.
        matrices.append(np.array([np.r_[system.roe_threshold*K.sum(axis=0)/12,-np.ones(n)]]))
        rhs.append(np.array([-system.roe_threshold*k0.sum()/12]))
        # IRR: necessary NPV test; sufficient for the supported one-sign-change flows.
        if not capital_free:
            matrices.append(np.array([np.r_[-capital_coeff@K,-discount]]))
            rhs.append(np.array([capital_coeff@k0]))
        low=np.r_[np.zeros(4),np.full(n,-np.inf)]; low[2]=ceil(lo)
        high=np.r_[ub,np.full(n,np.inf)]; high[2]=floor(hi)
        if capital_free:
            for j,w in zip((2,3),weights):
                if w>0: high[j]=0
            if np.any(low>high):
                checked+=1; upper_bounds.append(float('-inf')); continue
        res=milp(np.r_[np.zeros(4),-np.ones(n)],
            integrality=np.r_[np.ones(4),np.zeros(n)],bounds=Bounds(low,high),
            constraints=LinearConstraint(np.vstack(matrices),-np.inf,np.concatenate(rhs)),
            options={"mip_rel_gap":0.,"time_limit":remaining_time})
        checked+=1
        if res.status == 2:
            upper_bounds.append(float('-inf')); continue
        if res.status != 0: all_solved=False
        bound=getattr(res,'mip_dual_bound',None)
        upper_bounds.append(-float(bound)*unit if bound is not None and np.isfinite(bound) else float('inf'))
        if res.x is None: continue
        x=np.rint(res.x[:4])
        try:
            candidate=evaluate(x)
        except ValueError:
            warnings.append("Численный кандидат нарушил ограничения после округления; отброшен.")
            continue
        if candidate[2].accepted:
            if best is None or candidate[1].net_profit > best[1].net_profit:
                best=candidate
        elif candidate[2].status == 'irr_requires_review':
            warnings.append("Кандидат MILP имеет неопределённую/нетипичную IRR; он не допущен автоматически.")
    upper = max([0.]+upper_bounds) if all_solved and checked==len(intervals) else float('inf')
    if best is not None: upper=max(upper,best[1].net_profit)
    gap=max(0.,upper-best[1].net_profit) if best is not None and np.isfinite(upper) else None
    # Числовой допуск доказательства: 1 рубль или 1e-9 от прибыли.
    proven=bool(best and gap is not None and gap <= max(1.,abs(best[1].net_profit)*1e-9))
    no_solution=best is None and all_solved and checked==len(intervals) and upper<=0
    status='optimal' if proven else ('no_acceptable_allocation' if no_solution else 'incomplete')
    if status=='incomplete': warnings.append("Оптимальность не подтверждена: лимит времени, численные допуски или непроверенная IRR. Возвращён только проверенный допустимый кандидат, если он найден.")
    return OptimizationResult(*(best if best else (None,None,None)),status,proven,
        upper if np.isfinite(upper) else None,gap,perf_counter()-started,checked,len(intervals),tuple(dict.fromkeys(warnings)))

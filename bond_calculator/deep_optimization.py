"""Независимый многократный Differential Evolution; MILP не вызывается."""
from dataclasses import dataclass
from itertools import product
from math import isfinite
from time import perf_counter
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from .models import Deal, SystemParameters, add_months
from .channels import AllocationResult, calculate_allocation
from .returns import ReturnMetrics, calculate_returns
from .selection import better, better_values


@dataclass(frozen=True)
class DeepRun:
    seed: int
    seconds: float
    evaluations: int
    generations: int
    converged: bool
    verified_profit: float | None


@dataclass(frozen=True)
class DeepResult:
    volumes: tuple[int,int,int,int] | None
    allocation: AllocationResult | None
    metrics: ReturnMetrics | None
    status: str
    optimality_proven: bool
    elapsed_seconds: float
    runs: tuple[DeepRun,...]


class _Evaluator:
    """Ускоренный помесячный расчёт для тысяч оценок; итог проверяется оригиналом."""
    def __init__(self,d,s):
        self.d,self.s=d,s; self.unit=s.allocation_step; n=d.term_months
        self.upper=np.floor(np.array([min(d.institutional_demand,d.issue_volume),
            min(d.retail_demand,d.issue_volume),min(d.free_limit,d.issue_volume),
            min(d.free_limit,d.issue_volume)])/self.unit)
        self.limits=np.array([d.issue_volume,d.free_limit])/self.unit
        self.weights=np.array([s.resale_risk_weight,d.borrower_risk_weight])*s.capital_ratio
        if np.any(self.weights>1): raise ValueError('Капитал превышает позицию')
        rates=np.array([s.key_rate_at(add_months(d.placement_month,i)) for i in range(n)])
        self.funding=rates+s.funding_spread
        coupon=rates+d.coupon_spread if d.coupon_type=='floating' else np.full(n,d.fixed_coupon_rate)
        self.margin=coupon-self.funding-d.annual_pd*s.loss_given_default
        self.discount=np.exp(-np.arange(n+1)*np.log1p(s.irr_threshold)/12)
        self.opportunities=np.maximum(np.arange(n)-(s.resale_start_month-1),0)
        self.selling=np.arange(n)+1>=s.resale_start_month

    def details(self,x):
        d,s=self.d,self.s; q=d.monthly_resale_volume/self.unit
        position=np.maximum(x[2]-q*self.opportunities,0)
        sold=np.where(self.selling,np.minimum(position,q),0)
        capital=position*self.weights[0]+x[3]*self.weights[1]
        p=(position*(self.margin+self.weights[0]*self.funding)
           +x[3]*(self.margin+self.weights[1]*self.funding))/12+sold*s.resale_income_rate
        p[0]+=d.organizer_fee_rate*s.bank_placement_fee_share*(x[0]*(1-s.institutional_rebate_share)+x[1])+x[1]*s.retail_ecosystem_rate+(x[2]+x[3])*s.own_balance_placement_fee_rate
        net=p-s.tax_rate*np.maximum(p,0)
        flows=np.r_[-capital[0],net+capital-np.r_[capital[1:],0]]
        return net,capital,flows

    def objective(self,x): return -float(self.details(x)[0].sum())

    def constraints(self,x):
        net,capital,flows=self.details(x)
        profit=net.sum()
        if capital.sum()==0:
            roe_gap=npv=0.; shape=0.
        else:
            roe_gap=profit-self.s.roe_threshold*capital.sum()/12
            npv=flows@self.discount
            nz=flows[flows!=0]
            changes=np.count_nonzero(np.signbit(nz[1:])!=np.signbit(nz[:-1]))
            shape=0. if len(nz)>1 and nz[0]<0 and nz[-1]>0 and changes==1 else -1.
        return np.array([self.limits[0]-sum(x),self.limits[1]-x[2]-x[3],roe_gap,npv,shape])


def optimize_deep(d: Deal,s: SystemParameters,*,runs:int=4,seed:int=42,
                  maxiter:int=800,popsize:int=24,run_time_limit:float=20.) -> DeepResult:
    """Несколько seed, перенос лучшего решения и целочисленная доводка.

    Без доказательства оптимальности. Лимит времени проверяется между поколениями.
    MILP-результат не используется ни для инициализации, ни для улучшения.
    """
    if any(type(v) is not int or v<1 for v in (runs,maxiter,popsize)):
        raise ValueError('runs, maxiter и popsize должны быть положительными целыми')
    if not isfinite(run_time_limit) or run_time_limit<=0: raise ValueError('Некорректный лимит времени')
    start=perf_counter(); e=_Evaluator(d,s); best=None; history=[]
    def verified(x):
        if np.any(x<0) or np.any(x>e.upper) or np.any(e.constraints(x)<-1e-9): return None
        volumes=tuple(int(v)*e.unit for v in x)
        try: a=calculate_allocation(*volumes,d,s)
        except ValueError: return None
        m=calculate_returns(a,s)
        return (volumes,a,m) if m.accepted else None
    remaining=int(e.limits[0]); direct=np.zeros(4)
    fee=d.organizer_fee_rate*s.bank_placement_fee_share
    for j in sorted((0,1),key=lambda j: fee*(1-s.institutional_rebate_share) if j==0 else fee+s.retail_ecosystem_rate,reverse=True):
        direct[j]=min(e.upper[j],remaining); remaining-=int(direct[j])
    best=verified(direct)
    for run in range(runs):
        t=perf_counter(); rng=np.random.default_rng(seed+run)
        population=np.floor(rng.random((max(5,popsize*4),4))*(e.upper+1))
        population=np.minimum(population,e.upper)
        population[0]=direct; population[1]=0
        # Часть популяции изучает балансовые варианты при полном прямом спросе.
        population[2:len(population)//2,:2]=direct[:2]
        if best: population[2]=np.array(best[0])/e.unit
        res=differential_evolution(e.objective,list(zip(np.zeros(4),e.upper)),
            constraints=(NonlinearConstraint(e.constraints,0,np.inf),),integrality=True,
            init=population,seed=seed+run,maxiter=maxiter,popsize=popsize,tol=1e-10,
            atol=1e-10,polish=False,
            callback=lambda x,convergence: perf_counter()-t>=run_time_limit)
        candidates=[res.x]+list(res.population[np.isfinite(res.population_energies)])
        run_best=None
        for x in candidates:
            c=verified(np.rint(x))
            if c and better(c,run_best):run_best=c
        # Окрестность из 3^4 точек; не использует MILP.
        if run_best:
            x=np.array(run_best[0])//e.unit
            for _ in range(12):
                choices=[x+delta for delta in product((-1,0,1),repeat=4)]
                choices=[v for v in choices if np.all(v>=0) and np.all(v<=e.upper) and np.all(e.constraints(v)>=0)]
                next_x=x
                net,capital,_=e.details(x)
                score=(net.sum()*e.unit,capital.sum()*e.unit/12)
                for v in choices:
                    net,capital,_=e.details(v)
                    new_score=(net.sum()*e.unit,capital.sum()*e.unit/12)
                    if better_values(*new_score,*score):next_x=v;score=new_score
                if np.array_equal(next_x,x):break
                x=next_x
            c=verified(x)
            if c and better(c,run_best):run_best=c
            if better(run_best,best):best=run_best
        history.append(DeepRun(seed+run,perf_counter()-t,int(res.nfev),int(res.nit),
            bool(res.success),run_best[1].net_profit if run_best else None))
    return DeepResult(*(best if best else (None,None,None)),
        'feasible' if best else 'no_feasible_found',False,perf_counter()-start,tuple(history))

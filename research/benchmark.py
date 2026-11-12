"""Exploratory profit-only benchmark; run python -m research.benchmark.

No IRR/ROE acceptance constraints. Money and objectives in million RUB.
MILP enumerates resale break intervals, NOT every allocation.
"""
import json
from pathlib import Path
from time import perf_counter
from dataclasses import replace
import numpy as np
import scipy
from scipy.optimize import milp, Bounds, LinearConstraint, minimize, differential_evolution
from bond_calculator import *


def experiment(d, s):
    start = perf_counter()
    n = d.term_months
    cap = min(d.free_limit, d.issue_volume) / 1e6
    ub = np.array([min(d.institutional_demand,d.issue_volume)/1e6,
                   min(d.retail_demand,d.issue_volume)/1e6,cap,cap])
    A = np.array([[1.,1,1,1],[0.,0,1,1]])
    limits = np.array([d.issue_volume/1e6,d.free_limit/1e6])
    constraint = LinearConstraint(A, -np.inf, limits)
    funcs = [institutional_placement,retail_placement,hold_to_maturity]
    coeff = np.column_stack([[m.profit_before_tax for m in f(1e6,d,s).months] for f in funcs])/1e6
    rate = np.array([s.key_rate_at(m.month) for m in hold_to_maturity(1e6,d,s).months])
    coupon = rate+d.coupon_spread if d.coupon_type=='floating' else np.full(n,d.fixed_coupon_rate)
    monthly = (coupon-(rate+s.funding_spread)-d.annual_pd*s.loss_given_default
               +s.resale_risk_weight*s.capital_ratio*(rate+s.funding_spread))/12
    q = d.monthly_resale_volume/1e6
    opportunities = np.maximum(np.arange(n)-(s.resale_start_month-1),0)
    def resale(x):
        opening = np.maximum(x-q*opportunities,0)
        sold = np.where(np.arange(n)+1>=s.resale_start_month,np.minimum(opening,q),0)
        p = opening*monthly+sold*s.resale_income_rate
        p[0] += x*s.own_balance_placement_fee_rate
        return p
    def profit(x):
        p = coeff @ x[[0,1,3]] + resale(x[2])
        return float(np.sum(p-s.tax_rate*np.maximum(p,0)))
    def feasible(x): return bool(np.all(x>=0) and np.all(x<=ub+1e-8) and np.all(A@x<=limits+1e-8))
    rng=np.random.default_rng(42)
    # Cross-check accelerated evaluator against the existing four functions.
    for _ in range(30):
        x=np.floor(rng.random(4)*ub)
        if not feasible(x): continue
        actual=calculate_allocation(*(x*1e6),d,s).net_profit/1e6
        assert abs(actual-profit(x))<1e-7,(actual,profit(x))
    prep=perf_counter()-start
    results=[]
    def record(method,t,x,**extra):
        x=np.floor(np.maximum(x,0)+1e-7)
        assert feasible(x),x
        actual=calculate_allocation(*(x*1e6),d,s).net_profit/1e6
        assert abs(actual-profit(x))<1e-7
        results.append(dict(method=method,seconds=perf_counter()-t,profit_million=actual,
                            allocation_million=x.tolist(),**extra))
    # On each interval, monthly profits are affine. Net profit=min(p,(1-tax)*p).
    # Hypograph z<=p,z<=(1-tax)*p is exact when maximizing sum(z).
    t=perf_counter(); best=(-np.inf,None); gaps=[]; intervals=0
    knots=sorted(set([0.,cap]+[k*q for k in range(1,n+1) if 0<k*q<cap]))
    for lo,hi in zip(knots[:-1],knots[1:]):
        if np.ceil(lo)>np.floor(hi): continue
        slope=(resale(hi)-resale(lo))/(hi-lo); intercept=resale(lo)-slope*lo
        P=np.column_stack([coeff[:,0],coeff[:,1],slope,coeff[:,2]])
        mat=np.vstack([np.column_stack([A,np.zeros((2,n))]),
                       np.column_stack([-P,np.eye(n)]),
                       np.column_stack([-(1-s.tax_rate)*P,np.eye(n)])])
        upper=np.r_[limits,intercept,(1-s.tax_rate)*intercept]
        low=np.r_[np.zeros(4),np.full(n,-np.inf)]; low[2]=np.ceil(lo)
        high=np.r_[ub,np.full(n,np.inf)]; high[2]=np.floor(hi)
        res=milp(np.r_[np.zeros(4),-np.ones(n)],integrality=np.r_[np.ones(4),np.zeros(n)],
                 bounds=Bounds(low,high),constraints=LinearConstraint(mat,-np.inf,upper),
                 options={'mip_rel_gap':0.0,'time_limit':10})
        assert res.success,res.message
        intervals+=1; gaps.append(float(res.mip_gap))
        x=np.rint(res.x[:4]); value=profit(x)
        if value>best[0]: best=(value,x)
    record('piecewise_MILP',t,best[1],intervals=intervals,max_mip_gap=max(gaps))
    for seed in [0,1,2]:
        t=perf_counter()
        res=differential_evolution(lambda x:-profit(x),list(zip(np.zeros(4),ub)),
             constraints=(constraint,),integrality=True,seed=seed,maxiter=200,popsize=12,
             tol=1e-8,polish=False)
        record('DE_'+str(seed),t,res.x,nfev=res.nfev,success=bool(res.success))
    t=perf_counter(); candidates=[]; successes=0
    for _ in range(10):
        x=rng.random(4)*ub
        x*=min(1.,*(limits/(A@x+1e-20)))
        res=minimize(lambda x:-profit(x),x,method='SLSQP',bounds=Bounds(np.zeros(4),ub),
                     constraints=(constraint,),options={'maxiter':300,'ftol':1e-9})
        successes+=int(res.success)
        rounded=np.floor(np.maximum(res.x,0)+1e-7)
        if feasible(rounded): candidates.append(rounded)
    record('SLSQP_10_starts_floor',t,max(candidates,key=profit),successful_runs=successes)
    t=perf_counter(); candidates=[np.zeros(4)]
    for _ in range(5000):
        x=np.floor(rng.random(4)*ub)
        if feasible(x): candidates.append(x)
    record('random_5000',t,max(candidates,key=profit),feasible_samples=len(candidates)-1)
    for r in results: r['gap_to_milp_percent']=100*(best[0]-r['profit_million'])/abs(best[0])
    return dict(deal=d.name,prep_seconds=prep,results=results)


if __name__=='__main__':
    s=load_system(); deals=list(load_deals().values())
    deals.append(replace(deals[3],name='Stress fixed coupon 8%',fixed_coupon_rate=.08))
    output={'scope':'profit only, NO IRR/ROE thresholds', 'scipy':scipy.__version__,'experiments':[]}
    for d in deals:
        result=experiment(d,s); output['experiments'].append(result)
        print(d.name,[(r['method'],round(r['seconds'],3),round(r['gap_to_milp_percent'],4)) for r in result['results']],flush=True)
    Path('research/results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')

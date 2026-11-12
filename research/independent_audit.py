"""Independent Decimal ledger, exhaustive small grids and fixed-x3 base search.

The reference ledger does not call production financial/helper functions.
Production imports are used only on the comparison side. Run from repo root:
python -m research.independent_audit
"""
from dataclasses import replace
from decimal import Decimal as D
from itertools import product
import json
from pathlib import Path
import random
import time
import numpy as np
from scipy.optimize import brentq, milp, Bounds, LinearConstraint
from bond_calculator import load_deals, load_system, calculate_allocation
from bond_calculator.returns import calculate_returns, calculate_irr
from bond_calculator.optimization import optimize_milp
from bond_calculator.deep_optimization import optimize_deep, _Evaluator

OUT = Path(__file__).parent / 'audit_results.json'
ZERO = D(0)


def ledger(x, d, s):
    """Stateful monthly Decimal book; no vectorized production formulas."""
    x = list(map(lambda v: D(str(v)), x))
    dec = lambda v: D(str(v))
    positions = [x[2], x[3]]
    rows = []
    for month in range(d.term_months):
        year, index = divmod(d.placement_month.year*12+d.placement_month.month-1+month, 12)
        key = [p for p in s.key_rates if (p.effective_from.year,p.effective_from.month) <= (year,index+1)][-1]
        funding = dec(key.annual_rate)+dec(s.funding_spread)
        coupon = dec(d.fixed_coupon_rate) if d.coupon_type == 'fixed' else dec(key.annual_rate)+dec(d.coupon_spread)
        row = dict.fromkeys(('opening_position sold_principal redeemed_principal closing_position '
                            'opening_capital closing_capital placement_fee institutional_rebate '
                            'ecosystem_income resale_income coupon_income funding_cost expected_loss capital_income').split(), ZERO)
        if month == 0:
            row['placement_fee'] = (x[0]+x[1])*dec(d.organizer_fee_rate)*dec(s.bank_placement_fee_share)+(x[2]+x[3])*dec(s.own_balance_placement_fee_rate)
            row['institutional_rebate'] = x[0]*dec(d.organizer_fee_rate)*dec(s.bank_placement_fee_share)*dec(s.institutional_rebate_share)
            row['ecosystem_income'] = x[1]*dec(s.retail_ecosystem_rate)
        for j in range(2):
            opening = positions[j]
            weight = dec(s.resale_risk_weight if j == 0 else d.borrower_risk_weight)
            capital = opening*weight*dec(s.capital_ratio)
            sold = min(opening,dec(d.monthly_resale_volume)) if j == 0 and month+1 >= s.resale_start_month else ZERO
            positions[j] -= sold
            redeemed = positions[j] if month+1 == d.term_months else ZERO
            positions[j] -= redeemed
            for k,v in dict(opening_position=opening, sold_principal=sold, redeemed_principal=redeemed,
                            closing_position=positions[j], opening_capital=capital,
                            closing_capital=positions[j]*weight*dec(s.capital_ratio),
                            coupon_income=opening*coupon/12, funding_cost=opening*funding/12,
                            expected_loss=opening*dec(d.annual_pd)*dec(s.loss_given_default)/12,
                            capital_income=capital*funding/12, resale_income=sold*dec(s.resale_income_rate)).items():
                row[k] += v
        p = sum(row[k] for k in ('placement_fee','ecosystem_income','resale_income','coupon_income','capital_income'))-sum(row[k] for k in ('institutional_rebate','funding_cost','expected_loss'))
        row.update(profit_before_tax=p, tax=max(p,ZERO)*dec(s.tax_rate), net_profit=p-max(p,ZERO)*dec(s.tax_rate))
        rows.append(row)
    return rows


def metrics(rows, s):
    profit = float(sum(r['net_profit'] for r in rows))
    years = float(sum(r['opening_capital'] for r in rows)/12)
    flows = np.array([-float(rows[0]['opening_capital'])]+[float(r['net_profit']+r['opening_capital']-r['closing_capital']) for r in rows])
    nz = flows[flows != 0]
    conventional = len(nz)>1 and nz[0]<0 and nz[-1]>0 and np.count_nonzero(nz[1:]*nz[:-1]<0)==1
    annual = None
    if conventional:
        # Root in discount factor, not production log-rate bisection.
        scaled = flows/max(abs(flows))
        fun = lambda discount: np.polynomial.polynomial.polyval(discount,scaled)
        hi = 1.
        while fun(hi)<0: hi *= 2
        root = brentq(fun,0,hi,xtol=1e-14)
        annual = root**(-12)-1
    roe = profit/years if years else None
    accepted = profit>0 and (years==0 or (roe+1e-9>=s.roe_threshold and annual is not None and annual+1e-9>=s.irr_threshold))
    return dict(profit=profit,capital_years=years,roe=roe,irr=annual,accepted=bool(accepted),flow_sum=float(flows.sum()))


def compare(x,d,s):
    ref = ledger(x,d,s)
    actual = calculate_allocation(*x,d,s)
    error = max(abs(float(row[k])-getattr(a,k)) for row,a in zip(ref,actual.months) for k in row)
    assert error < .01, (d.id,x,error)
    m = metrics(ref,s); a = calculate_returns(actual,s)
    assert abs(m['flow_sum']-m['profit'])<.01
    assert abs(sum(float(r['sold_principal']+r['redeemed_principal']) for r in ref)-x[2]-x[3])<.01
    assert m['accepted']==a.accepted, (x,m,a)
    if m['irr'] is not None: assert abs(m['irr']-a.irr.annual)<1e-7*max(1,abs(m['irr']))
    if m['roe'] is not None: assert abs(m['roe']-a.roe)<1e-9*max(1,abs(m['roe']))
    # Third implementation, fast DE evaluator, checked against Decimal too.
    net,capital,flows = _Evaluator(d,s).details(np.array(x)/s.allocation_step)
    assert max(abs(net*s.allocation_step-np.array([float(r['net_profit']) for r in ref])))<.01
    assert max(abs(capital*s.allocation_step-np.array([float(r['opening_capital']) for r in ref])))<.01
    return error


def fixed_resale_search(d,s):
    """Enumerate EVERY x3, solve 3 integer variables, using Decimal-ledger coefficients.

    Only for base deals: all pretax contributions nonnegative, hence tax linear.
    No intervals, auxiliary tax variables, production coefficient/NPV helpers.
    Same HiGHS backend is a stated limitation of this cross-check.
    """
    started=time.perf_counter(); unit=s.allocation_step
    def summary(x):
        rows=ledger(x,d,s)
        assert all(r['profit_before_tax']>=0 for r in rows)
        net=np.array([float(r['net_profit'])/unit for r in rows])
        k=np.array([float(r['opening_capital'])/unit for r in rows])
        flow=np.r_[-k[0],net+k-np.r_[k[1:],0]]
        npv=sum(v/(1+s.irr_threshold)**(i/12) for i,v in enumerate(flow))
        return np.array([net.sum(),net.sum()-s.roe_threshold*k.sum()/12,npv])
    coeff=np.array([summary([unit if j==i else 0 for j in range(4)]) for i in (0,1,3)]).T
    a=np.vstack([np.ones(3),[0,0,1],-coeff[1:]])
    upper=np.floor(np.array([d.institutional_demand,d.retail_demand,min(d.free_limit,d.issue_volume)])/unit)
    best=0.; volumes=None; bound=0.; count=0
    for resale in range(int(min(d.free_limit,d.issue_volume)//unit)+1):
        const=summary([0,0,resale*unit,0])
        rhs=np.r_[d.issue_volume/unit-resale,d.free_limit/unit-resale,const[1:]]
        res=milp(-coeff[0],integrality=np.ones(3),bounds=Bounds(np.zeros(3),upper),
                 constraints=LinearConstraint(a,-np.inf,rhs),options={'mip_rel_gap':0.})
        count+=1
        assert res.status in (0,2),res.message
        if res.status==0:
            ub=(const[0]-res.mip_dual_bound)*unit
            bound=max(bound,ub)
            value=(const[0]-res.fun)*unit
            if value>best:
                v=np.rint(res.x).astype(int)*unit
                best=value; volumes=[int(v[0]),int(v[1]),resale*unit,int(v[2])]
        if resale%2000==0: print('enumeration',d.id,resale,flush=True)
    m=metrics(ledger(volumes,d,s),s)
    assert m['accepted']
    return dict(profit=best,volumes=volumes,upper_bound=bound,points=count,seconds=time.perf_counter()-started,metrics=m)


def run():
    ds=load_deals(); s=load_system(); rng=random.Random(173)
    result={'seed':173,'ledger_cases':0,'max_monthly_error_rub':0.,'small_grids':[],'base':{},'stress':[],'de_independent':[]}
    def save(): OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def check(x,d,sys):
        err=compare(x,d,sys); result['ledger_cases']+=1
        result['max_monthly_error_rub']=max(result['max_monthly_error_rub'],err)
    for d in ds.values():
        r=optimize_milp(d,s); check(r.volumes,d,s)
        result['base'][d.id]={'production_volumes':r.volumes,'production_profit':r.allocation.net_profit,'status':r.status,'metrics':metrics(ledger(r.volumes,d,s),s)}
        for j in range(4):
            x=[0]*4; x[j]=min(100_000_000,int([d.institutional_demand,d.retail_demand,d.free_limit,d.free_limit][j]//s.allocation_step)*s.allocation_step)
            check(x,d,s)
    # Independent exhaustive finite-grid optimum, changed units, zero caps, taxes, sale boundaries.
    for case in range(32):
        unit=[1_000_000,100_000][case%2]
        sys=replace(s,allocation_step=unit,tax_rate=[0,.25,1][case%3],resale_start_month=[2,4,10][case%3],
                    capital_ratio=[0,.12,1][case%3],resale_risk_weight=[1.2,1,1][case%3])
        d=replace(ds['A'],issue_volume=5*unit,free_limit=(case%6)*unit,institutional_demand=2*unit,
                  retail_demand=unit,monthly_resale_volume=[0,.3,1,8][case%4]*unit,term_months=[1,5,15,25][case%4],
                  borrower_risk_weight=[0,.65,1][case%3],coupon_type='fixed',coupon_spread=None,
                  fixed_coupon_rate=[.08,.155,.25,.4][case%4])
        best=0.; count=0
        for x0 in product(range(6),repeat=4):
            x=tuple(v*unit for v in x0)
            if sum(x)>d.issue_volume or x[0]>d.institutional_demand or x[1]>d.retail_demand or x[2]+x[3]>d.free_limit:continue
            check(x,d,sys); count+=1
            m=metrics(ledger(x,d,sys),sys)
            if m['accepted']:best=max(best,m['profit'])
        r=optimize_milp(d,sys)
        got=r.allocation.net_profit if r.allocation else 0.
        assert abs(best-got)<.01,(case,best,got,r.status)
        assert r.status==('optimal' if best>0 else 'no_acceptable_allocation'),(case,r.status)
        result['small_grids'].append(dict(case=case,points=count,profit=best,status=r.status))
    for case in range(160):
        base=ds['ABCD'[case%4]]
        d=replace(base,term_months=rng.choice([1,2,8,14,26,60]),monthly_resale_volume=rng.choice([0,500_000,123_456_789,1e10]),annual_pd=rng.choice([0,.004,.08,1]))
        sys=replace(s,tax_rate=rng.choice([0,.25,1]),own_balance_placement_fee_rate=rng.choice([0,.01]),resale_start_month=rng.choice([2,3,100]))
        x=[rng.randint(0,100)*s.allocation_step for _ in range(4)]
        check(x,d,sys)
    save(); print('ledger and exhaustive passed',result['ledger_cases'],flush=True)
    # Polynomial root examples; the unique-root/nonconventional case is in audit_boundaries.
    examples=[[-100,230,-132],[-100,50,-100],[-100,200,-50,1],[-100,0,121],[-100,-10,121],[-100,90]]
    result['irr_examples']=[]
    for flows in examples:
        roots=np.polynomial.polynomial.polyroots(flows)
        rates=sorted(float(1/r.real-1) for r in roots if abs(r.imag)<1e-9 and r.real>0)
        prod=calculate_irr(flows)
        result['irr_examples'].append(dict(flows=flows,all_real_monthly_irrs=rates,production_status=prod.status))
        if prod.status=='unique':assert len(rates)==1 and abs(rates[0]-prod.monthly)<1e-10
    for d in ds.values():
        result['base'][d.id]['independent_search']=fixed_resale_search(d,s)
        assert abs(result['base'][d.id]['independent_search']['profit']-result['base'][d.id]['production_profit'])<1
        save()
    for base in ds.values():
        for name,d,sys in [
            ('demand_minus20pct_issue',replace(base,institutional_demand=base.issue_volume*.8*base.institutional_demand/(base.institutional_demand+base.retail_demand),retail_demand=base.issue_volume*.8*base.retail_demand/(base.institutional_demand+base.retail_demand)),s),
            ('pd_plus10pct',replace(base,annual_pd=base.annual_pd*1.1),s),
            ('resale_minus10pct',replace(base,monthly_resale_volume=base.monthly_resale_volume*.9),s),
            ('rates_plus1pp',base,replace(s,key_rates=tuple(replace(p,annual_rate=p.annual_rate+.01) for p in s.key_rates))),
            ('limit_minus10pct',replace(base,free_limit=base.free_limit*.9),s),
            ('riskweight_plus10pct',replace(base,borrower_risk_weight=base.borrower_risk_weight*1.1),s),
            ('coupon_minus10bp',replace(base,**({'fixed_coupon_rate':base.fixed_coupon_rate-.001} if base.coupon_type=='fixed' else {'coupon_spread':base.coupon_spread-.001})),s)]:
            r=optimize_milp(d,sys)
            if r.volumes:check(r.volumes,d,sys)
            result['stress'].append(dict(deal=base.id,scenario=name,status=r.status,profit=r.allocation.net_profit if r.allocation else None,volumes=r.volumes))
    save(); print('stress passed',flush=True)
    for d in ds.values():
        for seed in (301,991,1723):
            # Separate calls: absolutely no transfer between seeds.
            r=optimize_deep(d,s,runs=1,seed=seed,run_time_limit=20)
            if r.volumes:check(r.volumes,d,s)
            result['de_independent'].append(dict(deal=d.id,seed=seed,profit=r.allocation.net_profit if r.allocation else None,seconds=r.elapsed_seconds,status=r.status))
            save(); print('DE',d.id,seed,flush=True)
    tiny=optimize_milp(ds['A'],s,time_limit=1e-9)
    assert tiny.status=='incomplete' and not tiny.optimality_proven
    short=optimize_deep(ds['A'],s,runs=1,run_time_limit=1e-9)
    assert not short.optimality_proven and short.metrics.accepted
    result['timeout_checks']={'milp':tiny.status,'de':short.status,'de_seconds':short.elapsed_seconds}
    result['completed']=True; save(); print('AUDIT COMPLETE',flush=True)


if __name__=='__main__':run()

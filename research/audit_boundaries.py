"""Targeted audit evidence: manual rows, threshold edges and old-solution stress."""
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import numpy as np
from bond_calculator import load_deals,load_system,calculate_allocation
from bond_calculator.optimization import optimize_milp
from bond_calculator.returns import calculate_returns,calculate_irr
from research.independent_audit import ledger,metrics,compare


def run():
    ds=load_deals(); s=load_system(); out={'manual':{},'thresholds':[],'old_allocation_stress':[]}
    for channel in range(4):
        x=[0]*4; x[channel]=1_000_000_000
        rows=ledger(x,ds['A'],s)
        out['manual'][str(channel+1)]=[{k:str(v) for k,v in rows[i].items()}|{'month_number':i+1} for i in (0,1,7,35)]
        compare(x,ds['A'],s)
    # Holding x3 large enough and pace small enough to leave principal at maturity.
    d=replace(ds['A'],term_months=3,monthly_resale_volume=123_456_789)
    out['residual']=[{k:str(v) for k,v in r.items()} for r in ledger([0,0,1e9,0],d,s)]
    compare([0,0,1e9,0],d,s)
    for d in ds.values():
        r=optimize_milp(d,s); ref=metrics(ledger(r.volumes,d,s),s)
        for field,measured in [('roe_threshold',ref['roe']),('irr_threshold',ref['irr'])]:
            for delta in (-1e-8,0,1e-8):
                sys=replace(s,**{field:measured+delta})
                m=calculate_returns(calculate_allocation(*r.volumes,d,sys),sys)
                assert m.accepted==(delta<=0),(d.id,field,delta)
                out['thresholds'].append(dict(deal=d.id,field=field,delta=delta,accepted=m.accepted))
        for name,changed in [('PD +1%',replace(d,annual_pd=d.annual_pd*1.01)),('resale -1%',replace(d,monthly_resale_volume=d.monthly_resale_volume*.99))]:
            m=metrics(ledger(r.volumes,changed,s),s)
            out['old_allocation_stress'].append(dict(deal=d.id,scenario=name,**m))
    # Three sign changes can still produce exactly one IRR: do not mislabel as multiple roots.
    flows=[-100,200,-1000,1000]
    roots=np.polynomial.polynomial.polyroots(flows)
    out['unique_nonconventional']={'flows':flows,'positive_real_discount_roots':[float(r.real) for r in roots if abs(r.imag)<1e-9 and r.real>0],'status':calculate_irr(flows).status}
    assert len(out['unique_nonconventional']['positive_real_discount_roots'])==1
    # Calendar starts can change; sales can start after maturity; capital can equal position.
    for start in (date(2027,6,1),date(2027,12,1),date(2028,12,1)):
        compare([1e6,1e6,1e6,1e6],replace(ds['A'],placement_month=start),s)
    compare([0,0,1e6,1e6],replace(ds['A'],borrower_risk_weight=1),replace(s,capital_ratio=1,resale_risk_weight=1))
    errors=[]
    for name,action in [('negative_spread',lambda:replace(ds['A'],coupon_spread=-.01)),('negative_rate',lambda:replace(s.key_rates[0],annual_rate=-.01)),('capital_above_position',lambda:calculate_allocation(0,0,1e6,0,ds['A'],replace(s,resale_risk_weight=10))),('zero_step',lambda:replace(s,allocation_step=0))]:
        try:action()
        except ValueError:errors.append(name)
        else:raise AssertionError(name)
    out['invalid_inputs_rejected']=errors
    Path('research/audit_boundaries.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Boundary audit passed')


if __name__=='__main__':run()

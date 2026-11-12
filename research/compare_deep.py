"""python -m research.compare_deep"""
from dataclasses import asdict
import json
from pathlib import Path
from bond_calculator import load_deals,load_system
from bond_calculator.optimization import optimize_milp
from bond_calculator.deep_optimization import optimize_deep


def main():
    rows=[]; s=load_system()
    for d in load_deals().values():
        light=optimize_milp(d,s)
        deep=optimize_deep(d,s)
        row={'deal':d.name,'milp_seconds':light.elapsed_seconds,'deep_seconds':deep.elapsed_seconds,
             'milp_status':light.status,'deep_status':deep.status,
             'milp_profit':light.metrics.net_profit if light.metrics else None,
             'deep_profit':deep.metrics.net_profit if deep.metrics else None,
             'milp_volumes':light.volumes,'deep_volumes':deep.volumes,
             'deep_roe':deep.metrics.roe if deep.metrics else None,
             'deep_irr':deep.metrics.irr.annual if deep.metrics else None,
             'runs':[asdict(r) for r in deep.runs]}
        row['profit_gap']=row['milp_profit']-row['deep_profit'] if light.metrics and deep.metrics else None
        rows.append(row)
        Path('research/deep_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

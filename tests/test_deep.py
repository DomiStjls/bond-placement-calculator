from dataclasses import replace
import unittest
import numpy as np
from bond_calculator import load_deals,load_system,calculate_allocation,calculate_returns
from bond_calculator.deep_optimization import _Evaluator,optimize_deep


class DeepTests(unittest.TestCase):
    def test_fast_evaluator_matches_full_accounting(self):
        rng=np.random.default_rng(7); s=load_system()
        for d in load_deals().values():
            for system in (s,replace(s,capital_ratio=0,resale_start_month=4)):
                e=_Evaluator(d,system)
                for _ in range(25):
                    x=np.floor(rng.random(4)*e.upper)
                    x=np.floor(x*min(1,e.limits[0]/max(sum(x),1),e.limits[1]/max(x[2]+x[3],1)))
                    a=calculate_allocation(*(int(v)*e.unit for v in x),d,system)
                    m=calculate_returns(a,system)
                    net,k,flows=e.details(x)
                    np.testing.assert_allclose(net*e.unit,[v.net_profit for v in a.months],atol=1e-6)
                    np.testing.assert_allclose(k*e.unit,[v.opening_capital for v in a.months],atol=1e-6)
                    np.testing.assert_allclose(flows*e.unit,m.equity_cash_flows,atol=1e-6)
                    self.assertAlmostEqual(float(flows@e.discount)*e.unit,m.npv_at_irr_threshold,places=4)

    def test_valid_result_and_reproducible_seed(self):
        d=replace(load_deals()['A'],free_limit=0); s=load_system()
        a=optimize_deep(d,s,runs=2,maxiter=8,popsize=5)
        b=optimize_deep(d,s,runs=2,maxiter=8,popsize=5)
        self.assertEqual(a.volumes,b.volumes)
        self.assertTrue(a.metrics.accepted)
        self.assertFalse(a.optimality_proven)
        self.assertEqual(len(a.runs),2)

    def test_zero_deal_does_not_claim_infeasibility_proof(self):
        r=optimize_deep(replace(load_deals()['A'],issue_volume=0),load_system(),runs=1,maxiter=3,popsize=5)
        self.assertEqual(r.status,'no_feasible_found')
        self.assertIsNone(r.metrics)
        self.assertFalse(r.optimality_proven)


if __name__=='__main__':unittest.main()

from dataclasses import replace
from itertools import product
import unittest
from bond_calculator import load_deals,load_system,calculate_allocation,calculate_returns
from bond_calculator.models import RatePeriod
from bond_calculator.optimization import optimize_milp
from bond_calculator.deep_optimization import optimize_deep
from bond_calculator.selection import better_values


class CapitalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.d=replace(load_deals()['A'],issue_volume=4e6,free_limit=4e6,
            institutional_demand=0,retail_demand=2e6,term_months=1,
            coupon_type='fixed',coupon_spread=None,fixed_coupon_rate=0,
            annual_pd=0,organizer_fee_rate=0,borrower_risk_weight=.2)
        self.s=replace(load_system(),key_rates=(RatePeriod(self.d.placement_month,0.),),
            funding_spread=0,tax_rate=0,own_balance_placement_fee_rate=.01,
            roe_threshold=0,irr_threshold=0)

    def test_two_stage_matches_exhaustive_ties(self):
        pairs=[]
        for x in product(range(5),repeat=4):
            try:a=calculate_allocation(*(v*1e6 for v in x),self.d,self.s)
            except ValueError:continue
            m=calculate_returns(a,self.s)
            if m.accepted:pairs.append((a.net_profit,-m.capital_years))
        r=optimize_milp(self.d,self.s)
        self.assertEqual((r.allocation.net_profit,-r.metrics.capital_years),max(pairs))
        self.assertEqual(r.volumes,(0,2_000_000,0,2_000_000))
        self.assertTrue(r.capital_optimality_proven)

    def test_de_prefers_capital_among_found_ties(self):
        r=optimize_deep(self.d,self.s,runs=1,maxiter=80,popsize=10,run_time_limit=5)
        self.assertEqual(r.volumes,(0,2_000_000,0,2_000_000))
        self.assertFalse(r.optimality_proven)

    def test_profit_priority_and_float_noise(self):
        self.assertFalse(better_values(99.99,0,100,1000))
        self.assertTrue(better_values(101,10000,100,0))
        self.assertTrue(better_values(100+1e-10,0,100,1000))

    def test_tiny_timeout_does_not_certify_capital(self):
        r=optimize_milp(self.d,self.s,time_limit=1e-9)
        self.assertFalse(r.capital_optimality_proven)


if __name__=='__main__':unittest.main()

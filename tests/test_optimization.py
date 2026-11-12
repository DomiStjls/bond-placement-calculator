from dataclasses import replace
from itertools import product
import unittest
from bond_calculator import *
from bond_calculator.optimization import optimize_milp


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.d=load_deals()['A']; self.s=load_system()

    def compare_exhaustive(self,d,s):
        best=0
        for x in product(range(int(d.issue_volume/s.allocation_step)+1),repeat=4):
            try: a=calculate_allocation(*(v*s.allocation_step for v in x),d,s)
            except ValueError: continue
            if calculate_returns(a,s).accepted: best=max(best,a.net_profit)
        result=optimize_milp(d,s)
        if best>0:
            self.assertEqual(result.status,'optimal',result)
            self.assertTrue(result.metrics.accepted)
            self.assertAlmostEqual(result.allocation.net_profit,best,places=4)
        else:
            self.assertEqual(result.status,'no_acceptable_allocation',result)
        return result

    def test_small_grid_matches_exhaustive_under_thresholds(self):
        d=replace(self.d,issue_volume=4e6,free_limit=3e6,institutional_demand=2e6,
                  retail_demand=1e6,monthly_resale_volume=.7e6,term_months=5)
        self.compare_exhaustive(d,self.s)
        self.compare_exhaustive(replace(d,monthly_resale_volume=0),self.s)
        self.compare_exhaustive(replace(d,term_months=1),self.s)
        self.compare_exhaustive(d,replace(self.s,resale_start_month=4))

    def test_negative_margin_and_zero_capital(self):
        d=replace(self.d,issue_volume=4e6,free_limit=3e6,institutional_demand=2e6,
                  retail_demand=1e6,monthly_resale_volume=.7e6,term_months=4,
                  coupon_type='fixed',coupon_spread=None,fixed_coupon_rate=.08)
        self.compare_exhaustive(d,self.s)
        self.compare_exhaustive(replace(d,fixed_coupon_rate=.2),replace(self.s,capital_ratio=0))
        self.compare_exhaustive(replace(d,fixed_coupon_rate=.2,borrower_risk_weight=0),self.s)

    def test_zero_limit_zero_volume_and_impossible_thresholds(self):
        result=optimize_milp(replace(self.d,free_limit=0),self.s)
        self.assertEqual(result.status,'optimal')
        self.assertEqual(result.metrics.status,'accepted_no_capital')
        self.assertEqual(result.volumes[2:],(0,0))
        zero=optimize_milp(replace(self.d,issue_volume=0),self.s)
        self.assertEqual(zero.status,'no_acceptable_allocation')
        self.assertIsNone(zero.volumes)
        d=replace(self.d,issue_volume=4e6,free_limit=3e6,institutional_demand=0,retail_demand=0)
        self.compare_exhaustive(d,replace(self.s,roe_threshold=100,irr_threshold=100))

    def test_all_four_base_deals_are_verified(self):
        for d in load_deals().values():
            r=optimize_milp(d,self.s)
            self.assertEqual(r.status,'optimal',r)
            self.assertTrue(r.metrics.accepted)
            self.assertTrue(all(v%self.s.allocation_step==0 for v in r.volumes))
            self.assertAlmostEqual(calculate_returns(calculate_allocation(*r.volumes,d,self.s),self.s).roe,r.metrics.roe)

    def test_timeout_is_not_claimed_optimal(self):
        r=optimize_milp(self.d,self.s,time_limit=1e-9)
        self.assertEqual(r.status,'incomplete')
        self.assertFalse(r.optimality_proven)
        self.assertTrue(r.metrics.accepted)

    def test_nonconventional_relaxation_is_not_accepted_as_optimal(self):
        d=replace(self.d,issue_volume=2e6,free_limit=2e6,institutional_demand=0,
                  retail_demand=0,term_months=4,coupon_type='fixed',
                  coupon_spread=None,fixed_coupon_rate=.10)
        s=replace(self.s,own_balance_placement_fee_rate=.2,resale_start_month=10,
                  roe_threshold=0,irr_threshold=0)
        r=optimize_milp(d,s)
        self.assertEqual(r.status,'incomplete')
        self.assertFalse(r.optimality_proven)
        self.assertIsNone(r.metrics)
        self.assertGreater(r.upper_bound_profit,0)


if __name__=='__main__': unittest.main()

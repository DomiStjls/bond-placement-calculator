from dataclasses import replace
import unittest
from bond_calculator import *


class ReturnsTests(unittest.TestCase):
    def setUp(self):
        self.d=load_deals()['A']; self.s=load_system()

    def test_irr_known_rates_and_zero(self):
        self.assertAlmostEqual(calculate_irr([-100,110]).monthly,.1)
        self.assertAlmostEqual(calculate_irr([-100,110]).annual,1.1**12-1)
        self.assertAlmostEqual(calculate_irr([-100]+[0]*11+[110]).annual,.1)
        self.assertAlmostEqual(calculate_irr([-100,100]).annual,0)
        self.assertLess(calculate_irr([-100,90]).annual,0)
        self.assertEqual(calculate_irr([-100,-10,121]).status,'unique')

    def test_nonconventional_no_root_invalid(self):
        self.assertEqual(calculate_irr([-100,230,-132]).status,'nonconventional')
        self.assertEqual(calculate_irr([0,10,20]).status,'no_root')
        with self.assertRaises(ValueError): calculate_irr([float('nan'),1])

    def test_no_capital_exception_and_no_profit(self):
        r=calculate_returns(calculate_allocation(0,1e9,0,0,self.d,self.s),self.s)
        self.assertTrue(r.accepted); self.assertIsNone(r.roe)
        self.assertEqual(r.irr.status,'no_capital')
        self.assertIsNone(r.irr_passed)
        self.assertTrue(r.warnings)
        zero=calculate_returns(calculate_allocation(0,0,0,0,self.d,self.s),self.s)
        self.assertFalse(zero.accepted)

    def test_capital_return_and_roe(self):
        a=calculate_allocation(0,0,1e9,0,self.d,self.s)
        r=calculate_returns(a,self.s)
        self.assertAlmostEqual(sum(r.equity_cash_flows),a.net_profit)
        self.assertEqual(r.equity_cash_flows[0],-144e6)
        self.assertAlmostEqual(r.roe,a.net_profit/(sum(m.opening_capital for m in a.months)/12))
        self.assertIsNone(r.monthly_annualized_roe[-1])
        self.assertGreater(r.equity_cash_flows[2],a.months[1].net_profit)

    def test_threshold_boundary_and_independent_failure(self):
        a=calculate_allocation(0,0,1e9,0,self.d,self.s)
        r=calculate_returns(a,self.s)
        exact=replace(self.s,roe_threshold=r.roe,irr_threshold=r.irr.annual)
        self.assertTrue(calculate_returns(a,exact).accepted)
        self.assertFalse(calculate_returns(a,replace(exact,roe_threshold=r.roe+.001)).accepted)
        self.assertFalse(calculate_returns(a,replace(exact,irr_threshold=r.irr.annual+.001)).accepted)


if __name__=='__main__': unittest.main()

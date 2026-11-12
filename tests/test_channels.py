from dataclasses import replace
from datetime import date
import math
import unittest

from bond_calculator import (
    RatePeriod, load_deals, load_system, institutional_placement, retail_placement,
    balance_for_resale, hold_to_maturity, calculate_allocation,
)


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.deals = load_deals()
        self.deal = self.deals["A"]
        self.system = load_system()

    def test_four_deals_and_units(self):
        self.assertEqual(list(self.deals), ["A", "B", "C", "D"])
        self.assertEqual(self.deal.issue_volume, 20_000_000_000)
        self.assertEqual(self.deals["D"].fixed_coupon_rate, 0.155)
        self.assertEqual(self.deals["C"].annual_pd, 0.008)

    def test_institutional_manual_billion(self):
        result = institutional_placement(1e9, self.deal, self.system)
        self.assertAlmostEqual(result.months[0].placement_fee, 1_250_000)
        self.assertAlmostEqual(result.months[0].institutional_rebate, 1_125_000)
        self.assertAlmostEqual(result.profit_before_tax, 125_000)
        self.assertAlmostEqual(result.standalone_net_profit, 93_750)
        self.assertTrue(all(m.net_profit == 0 for m in result.months[1:]))
        self.assertTrue(all(m.opening_capital == 0 for m in result.months))

    def test_retail_manual_billion(self):
        result = retail_placement(1e9, self.deal, self.system)
        self.assertAlmostEqual(result.profit_before_tax, 11_250_000)
        self.assertAlmostEqual(result.standalone_net_profit, 8_437_500)

    def test_balance_first_month_manual(self):
        m = hold_to_maturity(1e9, self.deal, self.system).months[0]
        self.assertAlmostEqual(m.opening_capital, 78_000_000)
        self.assertAlmostEqual(m.coupon_income, 1e9 * 0.158 / 12)
        self.assertAlmostEqual(m.funding_cost, 1e9 * 0.142 / 12)
        self.assertAlmostEqual(m.expected_loss, 150_000)
        self.assertAlmostEqual(m.capital_income, 923_000)
        self.assertAlmostEqual(m.profit_before_tax, 2_106_333.333333333, places=5)
        self.assertAlmostEqual(m.net_profit, 1_579_750, places=5)
        self.assertEqual(m.placement_fee, 0)

    def test_resale_calendar_and_remainder(self):
        deal = replace(self.deal, term_months=3)
        result = balance_for_resale(1e9, deal, self.system)
        self.assertEqual([m.sold_principal for m in result.months], [0, 4e8, 4e8])
        self.assertEqual([m.opening_position for m in result.months], [1e9, 1e9, 6e8])
        self.assertEqual(result.months[-1].redeemed_principal, 2e8)
        self.assertEqual(result.months[-1].closing_capital, 0)
        self.assertEqual(result.months[0].opening_capital, 144e6)
        self.assertAlmostEqual(sum(m.resale_income for m in result.months), 8e8 * 0.0054)
        self.assertEqual(sum(m.sold_principal + m.redeemed_principal for m in result.months), 1e9)
        # Бумаги, проданные в конце месяца 2, дают полный купон этого месяца.
        self.assertEqual(result.months[0].coupon_income, result.months[1].coupon_income)

    def test_exhausted_position_has_no_more_income(self):
        deal = replace(self.deal, term_months=4)
        result = balance_for_resale(5e8, deal, self.system)
        self.assertEqual([m.sold_principal for m in result.months], [0, 4e8, 1e8, 0])
        self.assertEqual(result.months[-1].opening_position, 0)
        self.assertEqual(result.months[-1].net_profit, 0)
        self.assertEqual(result.months[-1].redeemed_principal, 0)

    def test_zero_resale_and_one_month_term(self):
        for deal in (replace(self.deal, term_months=3, monthly_resale_volume=0),
                     replace(self.deal, term_months=1)):
            result = balance_for_resale(1e9, deal, self.system)
            self.assertTrue(all(m.sold_principal == 0 for m in result.months))
            self.assertEqual(result.months[-1].redeemed_principal, 1e9)
            self.assertEqual(result.months[-1].resale_income, 0)

    def test_hold_principal_is_not_profit(self):
        deal = replace(self.deal, term_months=2, annual_pd=0, coupon_spread=0)
        system = replace(self.system, funding_spread=0, capital_ratio=0)
        result = hold_to_maturity(1e9, deal, system)
        self.assertEqual(result.profit_before_tax, 0)
        self.assertEqual(result.months[-1].redeemed_principal, 1e9)

    def test_rates_change_first_day_and_fixed_coupon_stays(self):
        self.assertEqual(self.system.key_rate_at(date(2027, 6, 1)), 0.14)
        self.assertEqual(self.system.key_rate_at(date(2027, 7, 1)), 0.135)
        self.assertEqual(self.system.key_rate_at(date(2028, 1, 1)), 0.12)
        self.assertEqual(self.system.key_rate_at(date(2031, 1, 1)), 0.11)
        floating = hold_to_maturity(1e9, self.deal, self.system)
        self.assertAlmostEqual(floating.months[6].coupon_income - floating.months[7].coupon_income,
                               1e9 * 0.005 / 12)
        fixed = hold_to_maturity(1e9, self.deals["D"], self.system)
        self.assertEqual(fixed.months[0].coupon_income, fixed.months[-1].coupon_income)
        self.assertGreater(fixed.months[0].funding_cost, fixed.months[-1].funding_cost)

    def test_tax_after_monthly_aggregation_and_no_loss_carry(self):
        deal = replace(self.deal, term_months=2, coupon_type="fixed",
                       coupon_spread=None, fixed_coupon_rate=0, annual_pd=0)
        system = replace(self.system, funding_spread=0, capital_ratio=0,
                         key_rates=(RatePeriod(date(2026, 12, 1), 0.12),))
        result = calculate_allocation(0, 1e9, 0, 1e9, deal, system)
        self.assertAlmostEqual(result.months[0].profit_before_tax, 1_250_000)
        self.assertAlmostEqual(result.months[0].tax, 312_500)
        self.assertEqual(result.months[1].tax, 0)
        self.assertEqual(result.months[1].net_profit, -10_000_000)
        self.assertNotEqual(result.net_profit, sum(c.standalone_net_profit for c in result.channels))

    def test_zero_volumes_for_every_deal_and_channel(self):
        for deal in self.deals.values():
            for channel in (institutional_placement, retail_placement, balance_for_resale, hold_to_maturity):
                result = channel(0, deal, self.system)
                self.assertEqual(len(result.months), deal.term_months)
                self.assertEqual(result.standalone_net_profit, 0)
                self.assertEqual(result.profit_before_tax, 0)

    def test_all_four_deals_conserve_principal(self):
        for deal in self.deals.values():
            for channel in (balance_for_resale, hold_to_maturity):
                result = channel(1e9, deal, self.system)
                self.assertAlmostEqual(sum(m.sold_principal + m.redeemed_principal for m in result.months), 1e9)
                for m in result.months:
                    self.assertAlmostEqual(m.opening_position - m.sold_principal - m.redeemed_principal,
                                           m.closing_position)
                    self.assertGreaterEqual(m.closing_position, 0)
                    self.assertLessEqual(m.sold_principal, deal.monthly_resale_volume)

    def test_invalid_inputs_and_shared_constraints(self):
        for x in (-1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                hold_to_maturity(x, self.deal, self.system)
        with self.assertRaises(ValueError):
            retail_placement(2e9, self.deal, self.system)
        with self.assertRaises(ValueError):
            calculate_allocation(0, 0, 5e9, 5e9, self.deal, self.system)
        with self.assertRaises(ValueError):
            calculate_allocation(0, 0, 1_000_001, 0, self.deal, self.system)
        with self.assertRaises(ValueError):
            calculate_allocation(6e9, 1e9, 7e9, 7e9, replace(self.deal, free_limit=20e9), self.system)
        with self.assertRaises(ValueError):
            replace(self.system, tax_rate=25)
        with self.assertRaises(ValueError):
            replace(self.deal, term_months=0)
        with self.assertRaises(ValueError):
            replace(self.system, key_rates=tuple(reversed(self.system.key_rates)))
        with self.assertRaises(ValueError):
            replace(self.deal, placement_month=date(2026, 12, 2))


if __name__ == "__main__":
    unittest.main()

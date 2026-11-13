from dataclasses import replace
from unittest.mock import patch
import unittest
from bond_calculator import load_deals,load_system
from bond_calculator.optimization import optimize_milp
from bond_calculator.explanations import explain_distribution


class ExplanationTests(unittest.TestCase):
    def test_base_explanations_use_actual_constraints_and_reoptimization(self):
        d=load_deals()['A'];s=load_system();r=optimize_milp(d,s)
        e=explain_distribution(r,d,s)
        self.assertIn('Спрос розницы использован полностью.',e['notes'])
        self.assertIn('спрос розницы',e['probes'][1]['reason'])
        self.assertIn('ROE ниже порога',e['probes'][3]['reason'])
        retail=e['effects'][0]
        again=optimize_milp(replace(d,retail_demand=d.retail_demand*1.1),s)
        self.assertAlmostEqual(retail['delta'],again.allocation.net_profit-r.allocation.net_profit)
        self.assertTrue(retail['confirmed'])

    def test_incomplete_comparisons_do_not_rank_as_proven(self):
        d=load_deals()['A'];s=load_system();r=optimize_milp(d,s)
        unfinished=replace(r,optimality_proven=False,status='incomplete')
        with patch('bond_calculator.explanations.optimize_milp',return_value=unfinished):
            e=explain_distribution(unfinished,d,s)
        self.assertTrue(all(not row['confirmed'] for row in e['effects']))
        self.assertIn('не определено',e['ranking'])

    def test_zero_parameter_gets_explicit_one_step_change(self):
        d=replace(load_deals()['A'],free_limit=0,monthly_resale_volume=0)
        s=load_system();r=optimize_milp(d,s);e=explain_distribution(r,d,s)
        self.assertEqual(e['effects'][1]['after'],s.allocation_step)
        self.assertEqual(e['effects'][2]['after'],s.allocation_step)
        self.assertEqual(d.free_limit,0)


if __name__=='__main__':unittest.main()

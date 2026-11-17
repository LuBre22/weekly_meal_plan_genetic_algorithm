import math
import sys
import unittest
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diet_ga.constraints import MacroConstraints, Macros
from diet_ga.meals import Meal, MealCatalog


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.catalog = MealCatalog((
            Meal(1, "Breakfast", 100, 10, 5, 15, ("breakfast",)),
            Meal(2, "Lunch", 200, 20, 10, 30, ("lunch",)),
            Meal(3, "Dinner", 300, 30, 15, 45, ("dinner",)),
            Meal(4, "Extra protein", 300, 31, 15, 45, ("dinner",)),
            Meal(5, "Low fat", 300, 100, 14, 45, ("dinner",)),
        ))
        self.plan = ((1, 2, 3),) * 7
        self.constraints = MacroConstraints(Macros(60, 30, 90), Macros(427, 210, 630), 600, 4200)

    def test_exact_bounds_and_protein_objective(self):
        assessment = self.constraints.assess(self.plan, self.catalog)
        self.assertTrue(assessment.feasible)
        self.assertEqual(assessment.score, 420)
        improved = self.constraints.assess(((1, 2, 4),) * 7, self.catalog)
        self.assertTrue(improved.feasible)
        self.assertEqual(improved.score, 427)
        self.assertGreater(improved.score, assessment.score)

    def test_daily_violation_not_hidden_by_weekly_total(self):
        plan = ((1, 2, 5),) + self.plan[1:]
        constraints = MacroConstraints(Macros(60, 30, 90), Macros(1000, 1000, 1000), 0, 10000)
        assessment = constraints.assess(plan, self.catalog)
        self.assertFalse(assessment.feasible)
        self.assertEqual([(v.period, v.macro, v.kind, v.amount) for v in assessment.violations],
                         [("Monday", "fat_g", "minimum", 1)])
        self.assertLess(assessment.score, constraints.assess(self.plan, self.catalog).score)

    def test_weekly_upper_bound_and_infeasible_ranking(self):
        constraints = MacroConstraints(Macros(0, 0, 0), Macros(420, 210, 630), 0, 10000)
        low = constraints.assess(self.plan[:-1] + ((1, 2, 4),), self.catalog)
        high = constraints.assess(self.plan[:-2] + ((1, 2, 4),) * 2, self.catalog)
        self.assertEqual(low.violations[0].amount, 1)
        self.assertEqual(low.violations[0].kind, "maximum")
        self.assertGreater(low.score, high.score)
        self.assertGreater(constraints.assess(self.plan, self.catalog).score, low.score)

    def test_invalid_and_incompatible_limits(self):
        for minimum, maximum in (
            (Macros(-1, 0, 0), Macros(10, 10, 10)),
            (Macros(0, math.nan, 0), Macros(10, 10, 10)),
            (Macros(0, 0, 0), Macros(math.inf, 10, 10)),
            (Macros(2, 0, 0), Macros(13, 10, 10)),
        ):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    MacroConstraints(minimum, maximum, 0, 10000)

    def test_calorie_minimum_and_maximum_affect_fitness(self):
        baseline = MacroConstraints(Macros(0, 0, 0), Macros(1000, 1000, 1000), 0, 5000)
        daily = MacroConstraints(Macros(0, 0, 0), Macros(1000, 1000, 1000), 601, 5000)
        weekly = MacroConstraints(Macros(0, 0, 0), Macros(1000, 1000, 1000), 0, 4199)
        valid = baseline.assess(self.plan, self.catalog)
        low = daily.assess(self.plan, self.catalog)
        high = weekly.assess(self.plan, self.catalog)
        self.assertTrue(valid.feasible)
        self.assertEqual(len(low.violations), 7)
        self.assertEqual((low.violations[0].period, low.violations[0].macro,
                          low.violations[0].kind, low.violations[0].amount),
                         ("Monday", "calories", "minimum", 1))
        self.assertEqual((high.violations[0].period, high.violations[0].macro,
                          high.violations[0].kind, high.violations[0].amount),
                         ("Week", "calories", "maximum", 1))
        self.assertGreater(valid.score, low.score)
        self.assertGreater(valid.score, high.score)

    def test_invalid_calorie_limits(self):
        macros = (Macros(0, 0, 0), Macros(1000, 1000, 1000))
        for minimum, maximum in ((-1, 100), (0, math.nan), (math.inf, 100),
                                 (100, 699)):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    MacroConstraints(*macros, minimum, maximum)


if __name__ == "__main__":
    unittest.main()

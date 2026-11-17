import json
import random
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diet_ga.constraints import Violation
from diet_ga.meals import Meal, MealCatalog
from diet_ga.population import initialize_population
from diet_ga.report import format_report, run_with_report, weekly_report
from diet_ga.selection import ScoredPlan, evaluate


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.catalog = MealCatalog((
            Meal(1, "Breakfast A", 100, 10, 2, 20, ("breakfast",)),
            Meal(2, "Lunch A", 200, 20, 4, 30, ("lunch",)),
            Meal(3, "Dinner A", 300, 30, 6, 40, ("dinner",)),
            Meal(4, "Breakfast B", 500, 50, 10, 50, ("breakfast",)),
        ))
        self.plan = ((1, 2, 3),) * 6 + ((4, 2, 3),)
        self.options = dict(
            generations=5, min_exploration_generations=1,
            max_exploration_generations=3, stagnation_generations=2,
            improvement_threshold=0, exploration_mutation_rate=0.8,
            exploitation_mutation_rate=0.1, elite_count=1,
            immigrant_count=1, tournament_size=2,
        )

    def fitness(self, plan):
        return sum(self.catalog.by_id[meal].protein_g for day in plan for meal in day)

    def test_weekly_report_has_expected_days_meals_totals_and_average(self):
        report = weekly_report(ScoredPlan(self.plan, 123), self.catalog)
        self.assertEqual(report.score, 123)
        self.assertEqual([day.day for day in report.days],
                         ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        self.assertEqual(report.days[0].breakfast.name, "Breakfast A")
        self.assertEqual(report.days[-1].breakfast.id, 4)
        self.assertEqual(asdict(report.days[0].total),
                         dict(calories=600, protein_g=60, fat_g=12, carbs_g=90))
        self.assertEqual(asdict(report.weekly_total),
                         dict(calories=4600, protein_g=460, fat_g=92, carbs_g=660))
        for field, value in asdict(report.weekly_total).items():
            self.assertAlmostEqual(getattr(report.daily_average, field), value / 7)
        self.assertIsNone(report.statistics)
        rendered = format_report(report)
        self.assertIn("Sunday", rendered)
        self.assertIn("Breakfast B", rendered)
        self.assertIn("Weekly total: 4600.0 kcal", rendered)
        self.assertIn("Daily average: 657.1 kcal", rendered)
        self.assertNotIn("Run statistics", rendered)
        self.assertEqual(json.loads(json.dumps(asdict(report)))["days"][0]["lunch"]["id"], 2)

    def test_violations_appear_directly_below_weekly_total(self):
        report = weekly_report(ScoredPlan(self.plan, -1), self.catalog)
        violations = (
            Violation("Monday", "calories", 600, 700, "minimum"),
            Violation("Tuesday", "calories", 650, 700, "minimum"),
            Violation("Monday", "protein_g", 55, 60, "minimum"),
            Violation("Week", "protein_g", 460, 450, "maximum"),
            Violation("Week", "calories", 4600, 4500, "maximum"),
        )
        lines = format_report(report, violations=violations).splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("Weekly total:"))
        self.assertEqual(lines[index + 2],
                         "Weekly excess: calories 100 kcal | protein 10 g | fat 0 g | carbs 0 g")
        self.assertEqual(lines[index + 1],
                         "Daily shortfalls (sum over week): calories 150 kcal | protein 5 g | fat 0 g | carbs 0 g")
        self.assertTrue(lines[index + 3].startswith("Daily average:"))
        self.assertNotIn("Daily shortfalls", format_report(report))

    def test_invalid_plans_fail_explicitly(self):
        for plan in (self.plan[:-1], ((1, 2),) * 7, ((999, 2, 3),) * 7, ((2, 1, 3),) * 7):
            with self.subTest(plan=plan):
                with self.assertRaises(ValueError):
                    weekly_report(ScoredPlan(plan, 1), self.catalog)

    def test_stats_capture_actual_generations_improvements_and_switch(self):
        population = [ScoredPlan(self.plan, score) for score in (1, 2)]
        scores = iter((3, 3, 4, 4, 5))

        def fake_generation(current, *args, **kwargs):
            return [ScoredPlan(self.plan, next(scores)), current[0]]

        with patch("diet_ga.strategy.next_generation", side_effect=fake_generation):
            report = run_with_report(population, self.catalog, self.fitness, random.Random(1),
                                     include_statistics=True, **self.options)
        stats = report.statistics
        self.assertIsNotNone(stats)
        self.assertEqual(stats.population_size, 2)
        self.assertEqual([entry.generation for entry in stats.generations], list(range(6)))
        self.assertEqual([entry.best_score for entry in stats.generations], [2, 3, 3, 4, 4, 5])
        self.assertEqual([entry.generation for entry in stats.improvement_points], [1, 3, 5])
        self.assertEqual([entry.best_score for entry in stats.improvement_points], [3, 4, 5])
        self.assertEqual([entry.phase for entry in stats.generations],
                         ["initial", "exploration", "exploration", "exploration", "exploitation", "exploitation"])
        self.assertEqual(stats.first_exploitation_generation, 4)
        self.assertEqual(stats.generations[0].mean_score, 1.5)
        self.assertIsNone(stats.generations[0].mutation_rate)
        self.assertEqual(stats.generations[1].mutation_rate, 0.8)
        self.assertEqual(stats.generations[4].mutation_rate, 0.1)
        self.assertEqual(stats.generations[1].unique_plans, 1)
        self.assertEqual(report.score, 5)
        rendered = format_report(report)
        self.assertIn("Generation 3: 4.000", rendered)
        self.assertIn("First exploitation generation: 4", rendered)
        self.assertIn("0 | initial | - | 2.000 | 1.500 | 1", rendered)

    def test_statistics_flag_does_not_change_results_or_random_state(self):
        population = [
            evaluate(plan, self.fitness)
            for plan in initialize_population(10, self.catalog, random.Random(10))
        ]
        rng1, rng2 = random.Random(5), random.Random(5)
        normal = run_with_report(population, self.catalog, self.fitness, rng1, **self.options)
        detailed = run_with_report(population, self.catalog, self.fitness, rng2,
                                   include_statistics=True, **self.options)
        self.assertIsNone(normal.statistics)
        self.assertIsNotNone(detailed.statistics)
        self.assertEqual(normal.score, detailed.score)
        self.assertEqual(normal.days, detailed.days)
        self.assertEqual(normal.weekly_total, detailed.weekly_total)
        self.assertEqual(rng1.getstate(), rng2.getstate())
        self.assertEqual(detailed.statistics.generations[-1].best_score, detailed.score)
        self.assertEqual(len(detailed.statistics.generations), self.options["generations"] + 1)
        self.assertEqual(
            detailed.statistics.generations[detailed.statistics.first_exploitation_generation].phase,
            "exploitation",
        )

    def test_immediate_exploitation_and_no_improvements(self):
        population = [ScoredPlan(self.plan, 1), ScoredPlan(self.plan, 1)]
        options = dict(self.options)
        options.update(min_exploration_generations=0, max_exploration_generations=0)
        with patch("diet_ga.strategy.next_generation", side_effect=lambda pop, *a, **kw: pop):
            report = run_with_report(population, self.catalog, self.fitness, random.Random(1),
                                     include_statistics=True, **options)
        self.assertEqual(report.statistics.first_exploitation_generation, 1)
        self.assertEqual(report.statistics.improvement_points, ())
        self.assertIn("    None", format_report(report))


if __name__ == "__main__":
    unittest.main()

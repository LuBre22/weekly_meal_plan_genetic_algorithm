import math
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diet_ga.meals import MEAL_TYPES, MealCatalog
from diet_ga.population import initialize_population
from diet_ga.selection import ScoredPlan, evaluate
from diet_ga.strategy import run_evolution


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.catalog = MealCatalog.from_csv(Path(__file__).resolve().parents[1] / "meals.csv")
        self.rng = random.Random(12)
        self.fitness = lambda plan: sum(
            self.catalog.by_id[meal_id].protein_g for day in plan for meal_id in day
        )
        self.population = [
            evaluate(plan, self.fitness)
            for plan in initialize_population(20, self.catalog, self.rng)
        ]
        self.options = dict(
            generations=10, min_exploration_generations=2,
            max_exploration_generations=7, stagnation_generations=3,
            improvement_threshold=0,
            exploration_mutation_rate=0.6, exploitation_mutation_rate=0.05,
            elite_count=2, immigrant_count=3, tournament_size=3,
        )

    def run_strategy(self, **overrides):
        options = dict(self.options)
        options.update(overrides)
        return run_evolution(self.population, self.catalog, self.fitness, self.rng, **options)

    def test_stagnation_switches_after_minimum(self):
        with patch("diet_ga.strategy.next_generation", side_effect=lambda pop, *a, **kw: pop) as next_gen:
            result = self.run_strategy(min_exploration_generations=4)
        self.assertEqual(result, self.population)
        self.assertEqual(
            [call.kwargs["mutation_rate"] for call in next_gen.call_args_list],
            [0.6] * 4 + [0.05] * 6,
        )
        self.assertTrue(all(call.kwargs["immigrant_count"] == 3 for call in next_gen.call_args_list))

    def test_maximum_forces_switch_despite_continuing_improvements(self):
        baseline = max(item.score for item in self.population)
        plan = self.population[0].plan
        scores = iter(range(1, 11))
        with patch("diet_ga.strategy.next_generation", side_effect=lambda *a, **kw: [
            ScoredPlan(plan, baseline + next(scores))
        ]) as next_gen:
            self.run_strategy()
        self.assertEqual(
            [call.kwargs["mutation_rate"] for call in next_gen.call_args_list],
            [0.6] * 7 + [0.05] * 3,
        )

    def test_small_gains_accumulate_and_reset_stagnation(self):
        baseline = max(item.score for item in self.population)
        plan = self.population[0].plan
        scores = iter([0.4, 1.0, 1.2, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])
        with patch("diet_ga.strategy.next_generation", side_effect=lambda *a, **kw: [
            ScoredPlan(plan, baseline + next(scores))
        ]) as next_gen:
            self.run_strategy(improvement_threshold=1.0)
        self.assertEqual(
            [call.kwargs["mutation_rate"] for call in next_gen.call_args_list],
            [0.6] * 5 + [0.05] * 5,
        )

    def test_zero_and_equal_bounds(self):
        for bound in (0, 4):
            with self.subTest(bound=bound):
                with patch("diet_ga.strategy.next_generation", side_effect=lambda pop, *a, **kw: pop) as next_gen:
                    self.run_strategy(min_exploration_generations=bound, max_exploration_generations=bound)
                self.assertEqual(
                    [call.kwargs["mutation_rate"] for call in next_gen.call_args_list],
                    [0.6] * bound + [0.05] * (10 - bound),
                )

    def test_full_run_is_reproducible_valid_and_preserves_best(self):
        first = self.run_strategy()
        second = run_evolution(
            self.population, self.catalog, self.fitness, random.Random(12), **self.options
        )
        third = run_evolution(
            self.population, self.catalog, self.fitness, random.Random(12), **self.options
        )
        self.assertEqual(second, third)
        self.assertEqual(len(first), len(self.population))
        self.assertGreaterEqual(
            max(item.score for item in first), max(item.score for item in self.population)
        )
        for item in first:
            self.assertEqual(item.score, self.fitness(item.plan))
            self.assertEqual(len(item.plan), 7)
            for day in item.plan:
                for meal_id, meal_type in zip(day, MEAL_TYPES):
                    self.assertIn(meal_id, self.catalog.eligible[meal_type])

    def test_invalid_schedule_rejected(self):
        for options in (
            {"generations": 0}, {"generations": 1.5}, {"generations": True},
            {"min_exploration_generations": -1},
            {"min_exploration_generations": 8},
            {"max_exploration_generations": 10},
            {"max_exploration_generations": 2.5},
            {"min_exploration_generations": True},
            {"stagnation_generations": 0},
            {"stagnation_generations": 1.5},
            {"improvement_threshold": -1},
            {"improvement_threshold": math.nan},
            {"improvement_threshold": math.inf},
            {"exploration_mutation_rate": -0.1},
            {"exploration_mutation_rate": 1.1},
            {"exploitation_mutation_rate": math.nan},
            {"exploitation_mutation_rate": 0.8},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.run_strategy(**options)

    def test_schedule_rejects_zero_elites(self):
        with self.assertRaisesRegex(ValueError, "Elite count must be a positive integer"):
            self.run_strategy(elite_count=0)


if __name__ == "__main__":
    unittest.main()

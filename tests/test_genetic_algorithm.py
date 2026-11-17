import csv
import hashlib
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diet_ga.crossover import crossover
from diet_ga.constraints import MacroConstraints, Macros
from diet_ga.evolution import next_generation
from diet_ga.meals import MEAL_TYPES, Meal, MealCatalog
from diet_ga.mutation import mutate
from diet_ga.population import initialize_population
from diet_ga.selection import ScoredPlan, evaluate, tournament_select


ROOT = Path(__file__).resolve().parents[1]


class GeneticAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.catalog = MealCatalog.from_csv(ROOT / "meals.csv")
        self.rng = random.Random(42)

    def fitness(self, plan):
        return sum(self.catalog.by_id[meal_id].protein_g for day in plan for meal_id in day)

    def assert_valid_plan(self, plan):
        self.assertEqual(len(plan), 7)
        for day in plan:
            self.assertEqual(len(day), 3)
            for meal_id, meal_type in zip(day, MEAL_TYPES):
                self.assertIn(meal_id, self.catalog.eligible[meal_type])

    def test_csv_has_stable_unique_ids_and_preserved_data(self):
        self.assertEqual(len(self.catalog.by_id), 200)
        self.assertEqual(self.catalog.by_id[1].name, "Greek yogurt and berries")
        self.assertEqual(self.catalog.by_id[30].protein_g, 39)
        self.assertEqual(self.catalog.by_id[9].meal_types, ("breakfast", "lunch"))
        with (ROOT / "meals.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(row["id"]) for row in rows], list(range(1, 201)))
        self.assertEqual(len({row["name"].casefold() for row in rows}), len(rows))
        original = [{key: value for key, value in row.items() if key != "id"} for row in rows[:30]]
        digest = hashlib.sha256(json.dumps(original, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, "b0316186c2d8eca7fea9a213bc5a8cc4aa1de8236fd9b997f6f52e15fd4e7b90")
        for meal in list(self.catalog.by_id.values())[30:]:
            expected = 4 * meal.protein_g + 9 * meal.fat_g + 4 * meal.carbs_g
            self.assertLessEqual(abs(meal.calories - expected), 15, meal.name)

    def test_expanded_catalog_can_meet_example_constraints(self):
        constraints = MacroConstraints(
            Macros(120, 60, 225), Macros(1400, 700, 2625), 1800, 16100,
        )
        plan = ((1, 81, 162),) * 7
        self.assert_valid_plan(plan)
        self.assertTrue(constraints.assess(plan, self.catalog).feasible)

    def test_initial_population_size_eligibility_and_reproducibility(self):
        plans = initialize_population(200, self.catalog, self.rng)
        self.assertEqual(len(plans), 200)
        self.assertEqual(plans, initialize_population(200, self.catalog, random.Random(42)))
        self.assertGreater(len(set(plans)), 1)
        for plan in plans:
            self.assert_valid_plan(plan)
        with self.assertRaises(ValueError):
            initialize_population(0, self.catalog, self.rng)

    def test_tournament_selects_best_of_sample(self):
        plans = initialize_population(5, self.catalog, self.rng)
        population = [ScoredPlan(plan, float(i)) for i, plan in enumerate(plans)]
        self.assertEqual(tournament_select(population, 5, self.rng), population[-1])
        for invalid in (0, 6):
            with self.assertRaises(ValueError):
                tournament_select(population, invalid, self.rng)

    def test_day_crossover_preserves_complete_days(self):
        first, second = initialize_population(2, self.catalog, self.rng)
        child = crossover(first, second, self.rng)
        self.assert_valid_plan(child)
        for i, day in enumerate(child):
            self.assertIn(day, (first[i], second[i]))
        with self.assertRaises(ValueError):
            crossover(first[:-1], second, self.rng)

    def test_mutation_zero_one_and_eligibility(self):
        plan = initialize_population(1, self.catalog, self.rng)[0]
        self.assertEqual(mutate(plan, self.catalog, 0, self.rng), plan)
        mutated = mutate(plan, self.catalog, 1, self.rng)
        self.assert_valid_plan(mutated)
        for before, after in zip(plan, mutated):
            self.assertEqual(sum(a != b for a, b in zip(before, after)), 1)
        for invalid in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                mutate(plan, self.catalog, invalid, self.rng)

    def test_mutation_with_only_one_eligible_meal_per_type(self):
        catalog = MealCatalog((
            Meal(1, "breakfast", 1, 1, 1, 1, ("breakfast",)),
            Meal(2, "lunch", 1, 1, 1, 1, ("lunch",)),
            Meal(3, "dinner", 1, 1, 1, 1, ("dinner",)),
        ))
        plan = initialize_population(1, catalog, self.rng)[0]
        self.assertEqual(mutate(plan, catalog, 1, self.rng), plan)

    def test_generation_preserves_elites_and_recalculates_all_other_scores(self):
        population = [
            evaluate(plan, self.fitness)
            for plan in initialize_population(30, self.catalog, self.rng)
        ]
        elites = sorted(population, key=lambda item: item.score, reverse=True)[:3]
        calls = []

        def fitness(plan):
            calls.append(plan)
            return self.fitness(plan)

        new = next_generation(
            population, self.catalog, fitness, self.rng,
            elite_count=3, immigrant_count=4, tournament_size=3, mutation_rate=1,
        )
        self.assertEqual(len(new), 30)
        self.assertEqual(new[:3], elites)
        self.assertEqual(len(calls), 27)
        self.assertEqual(calls, [item.plan for item in new[3:]])
        for item in new:
            self.assert_valid_plan(item.plan)
            self.assertEqual(item.score, self.fitness(item.plan))
        self.assertEqual(len(population), 30)

    def test_generation_uses_random_immigrants_and_adjustable_mutation(self):
        population = [
            evaluate(plan, self.fitness)
            for plan in initialize_population(10, self.catalog, self.rng)
        ]
        rng = random.Random(99)
        expected_immigrants = initialize_population(3, self.catalog, random.Random(99))
        new = next_generation(
            population, self.catalog, self.fitness, rng,
            elite_count=7, immigrant_count=3, tournament_size=3, mutation_rate=0,
        )
        self.assertEqual([item.plan for item in new[7:]], expected_immigrants)
        newer = next_generation(
            new, self.catalog, self.fitness, rng,
            elite_count=1, immigrant_count=2, tournament_size=3, mutation_rate=1,
        )
        self.assertEqual(len(newer), 10)
        self.assertTrue(all(item.score == self.fitness(item.plan) for item in newer))

    def test_multiple_generations_remain_valid_and_preserve_best(self):
        population = [
            evaluate(plan, self.fitness)
            for plan in initialize_population(50, self.catalog, self.rng)
        ]
        best = max(item.score for item in population)
        for generation in range(50):
            population = next_generation(
                population, self.catalog, self.fitness, self.rng,
                elite_count=2, immigrant_count=5, tournament_size=3,
                mutation_rate=0.1 if generation < 25 else 0.02,
            )
            self.assertEqual(len(population), 50)
            self.assertGreaterEqual(max(item.score for item in population), best)
            best = max(item.score for item in population)
            for item in population:
                self.assert_valid_plan(item.plan)
                self.assertEqual(item.score, self.fitness(item.plan))

    def test_one_elite_survives_even_when_other_slots_are_immigrants(self):
        population = [
            evaluate(plan, self.fitness)
            for plan in initialize_population(10, self.catalog, self.rng)
        ]
        best = max(population, key=lambda item: item.score)
        new = next_generation(
            population, self.catalog, self.fitness, self.rng,
            elite_count=1, immigrant_count=9, tournament_size=3, mutation_rate=1,
        )
        self.assertEqual(len(new), 10)
        self.assertIs(new[0], best)

    def test_elitism_is_required(self):
        population = [evaluate(plan, self.fitness) for plan in
                      initialize_population(2, self.catalog, self.rng)]
        for count in (0, -1, 0.5, True):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "Elite count must be a positive integer"):
                    next_generation(
                        population, self.catalog, self.fitness, self.rng,
                        elite_count=count, immigrant_count=0, tournament_size=1, mutation_rate=0,
                    )

    def test_invalid_inputs_are_explicit(self):
        plan = initialize_population(1, self.catalog, self.rng)[0]
        with self.assertRaises(ValueError):
            evaluate(plan, lambda _: float("nan"))
        population = [evaluate(plan, self.fitness)]
        with self.assertRaisesRegex(ValueError, "scores must be finite"):
            next_generation(
                [ScoredPlan(plan, math.nan)], self.catalog, self.fitness, self.rng,
                elite_count=1, immigrant_count=0, tournament_size=1, mutation_rate=0,
            )
        for kwargs in (
            dict(elite_count=1, immigrant_count=1, tournament_size=1, mutation_rate=0),
            dict(elite_count=0, immigrant_count=0, tournament_size=2, mutation_rate=0),
            dict(elite_count=1, immigrant_count=0, tournament_size=1, mutation_rate=math.nan),
        ):
            with self.assertRaises(ValueError):
                next_generation(population, self.catalog, self.fitness, self.rng, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("name,calories\nExample,100\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSV must include"):
                MealCatalog.from_csv(path)
        with self.assertRaisesRegex(ValueError, "Duplicate meal ID"):
            MealCatalog((self.catalog.by_id[1], self.catalog.by_id[1]))


if __name__ == "__main__":
    unittest.main()

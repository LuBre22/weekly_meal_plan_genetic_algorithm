"""Run an example optimization; adjust the settings below before running."""

import argparse
import random
from pathlib import Path

from diet_ga.constraints import MacroConstraints, Macros
from diet_ga.meals import MealCatalog, Plan
from diet_ga.population import initialize_population
from diet_ga.report import format_report, run_with_report
from diet_ga.selection import evaluate

# Input and reproducibility
MEALS_FILE = Path(__file__).resolve().parent / "meals.csv"
RANDOM_SEED = None  # Set to None for a different random sequence each run.
INCLUDE_STATISTICS = False

# Constraints: calories in kcal, macros in grams. Set to None to require CLI input.
DAILY_MIN_CALORIES = 1800
DAILY_MIN_PROTEIN = 120
DAILY_MIN_FAT = 60
DAILY_MIN_CARBS = 225
WEEKLY_MAX_PROTEIN = 1400
WEEKLY_MAX_FAT = 700
WEEKLY_MAX_CARBS = 2625
WEEKLY_MAX_CALORIES = 16100

# Optional variety rules: set a positive per-meal weekly limit to enable.
# None disables both the weekly limit and the same-day duplicate restriction.
MAX_MEAL_OCCURRENCES = 2

# Population and selection
POPULATION_SIZE = 200
ELITE_COUNT = 5
IMMIGRANT_COUNT = 10
TOURNAMENT_SIZE = 3

# Generations and adaptive phase switch
GENERATIONS = 100
MIN_EXPLORATION_GENERATIONS = 20
MAX_EXPLORATION_GENERATIONS = 70
STAGNATION_GENERATIONS = 10
IMPROVEMENT_THRESHOLD = 1.0
EXPLORATION_MUTATION_RATE = 0.5
EXPLOITATION_MUTATION_RATE = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description="Maximize weekly protein subject to nutrition constraints.")
    defaults = (
        ("daily-min-calories", DAILY_MIN_CALORIES),
        ("daily-min-protein", DAILY_MIN_PROTEIN),
        ("daily-min-fat", DAILY_MIN_FAT),
        ("daily-min-carbs", DAILY_MIN_CARBS),
        ("weekly-max-protein", WEEKLY_MAX_PROTEIN),
        ("weekly-max-fat", WEEKLY_MAX_FAT),
        ("weekly-max-carbs", WEEKLY_MAX_CARBS),
        ("weekly-max-calories", WEEKLY_MAX_CALORIES),
    )
    for name, default in defaults:
        unit = "kcal" if name.endswith("calories") else "grams"
        parser.add_argument(
            f"--{name}", type=float, default=default, required=default is None, metavar=unit.upper(),
            help=f"Override the corresponding setting in main.py ({unit})",
        )
    parser.add_argument(
        "--max-meal-occurrences", type=int, default=MAX_MEAL_OCCURRENCES, metavar="COUNT",
        help="Enable variety rules with a per-meal weekly maximum (default: disabled)",
    )
    args = parser.parse_args()
    try:
        constraints = MacroConstraints(
            daily_minimum=Macros(args.daily_min_protein, args.daily_min_fat, args.daily_min_carbs),
            weekly_maximum=Macros(args.weekly_max_protein, args.weekly_max_fat, args.weekly_max_carbs),
            daily_min_calories=args.daily_min_calories,
            weekly_max_calories=args.weekly_max_calories,
            max_meal_occurrences=args.max_meal_occurrences,
        )
    except ValueError as exc:
        parser.error(str(exc))
    catalog = MealCatalog.from_csv(MEALS_FILE)
    rng = random.Random(RANDOM_SEED)

    def fitness(plan: Plan) -> float:
        return constraints.assess(plan, catalog).score

    population = [
        evaluate(plan, fitness) for plan in initialize_population(POPULATION_SIZE, catalog, rng)
    ]
    report = run_with_report(
        population, catalog, fitness, rng,
        generations=GENERATIONS,
        min_exploration_generations=MIN_EXPLORATION_GENERATIONS,
        max_exploration_generations=MAX_EXPLORATION_GENERATIONS,
        stagnation_generations=STAGNATION_GENERATIONS,
        improvement_threshold=IMPROVEMENT_THRESHOLD,
        exploration_mutation_rate=EXPLORATION_MUTATION_RATE,
        exploitation_mutation_rate=EXPLOITATION_MUTATION_RATE,
        elite_count=ELITE_COUNT,
        immigrant_count=IMMIGRANT_COUNT,
        tournament_size=TOURNAMENT_SIZE,
        include_statistics=INCLUDE_STATISTICS,
    )
    plan: Plan = tuple((day.breakfast.id, day.lunch.id, day.dinner.id) for day in report.days)
    assessment = constraints.assess(plan, catalog)
    if assessment.feasible:
        print("Constraints satisfied; maximizing weekly protein.")
    else:
        print("No feasible plan found in this run; showing the least-violating candidate.")
    print(format_report(report, violations=assessment.violations, variety=assessment.variety))
    return 0 if assessment.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())

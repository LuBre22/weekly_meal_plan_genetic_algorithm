"""Structured and human-readable reports for an evolution run."""

import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from .meals import DAYS_PER_WEEK, MEAL_TYPES, Meal, MealCatalog
from .selection import Fitness, ScoredPlan
from .strategy import run_evolution

if TYPE_CHECKING:
    from .constraints import VarietyViolations, Violation

DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class Nutrition:
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float


@dataclass(frozen=True)
class DailyReport:
    day: str
    breakfast: Meal
    lunch: Meal
    dinner: Meal
    total: Nutrition


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    phase: str
    mutation_rate: Optional[float]
    best_score: float
    mean_score: float
    unique_plans: int


@dataclass(frozen=True)
class RunStatistics:
    population_size: int
    generations: Tuple[GenerationStats, ...]
    improvement_points: Tuple[GenerationStats, ...]
    first_exploitation_generation: int


@dataclass(frozen=True)
class WeeklyReport:
    score: float
    days: Tuple[DailyReport, ...]
    weekly_total: Nutrition
    daily_average: Nutrition
    statistics: Optional[RunStatistics] = None


def _nutrition(meals: Sequence[Meal]) -> Nutrition:
    return Nutrition(
        sum(meal.calories for meal in meals),
        sum(meal.protein_g for meal in meals),
        sum(meal.fat_g for meal in meals),
        sum(meal.carbs_g for meal in meals),
    )


def weekly_report(candidate: ScoredPlan, catalog: MealCatalog) -> WeeklyReport:
    """Resolve and summarize a scored plan without recalculating its fitness."""
    if not math.isfinite(candidate.score):
        raise ValueError("Fitness must be a finite number")
    if len(candidate.plan) != DAYS_PER_WEEK:
        raise ValueError("A weekly plan must contain seven days")
    days = []
    all_meals = []
    for day_name, day in zip(DAY_NAMES, candidate.plan):
        if len(day) != len(MEAL_TYPES):
            raise ValueError(f"{day_name} must contain three meals")
        meals = []
        for meal_id, meal_type in zip(day, MEAL_TYPES):
            if meal_id not in catalog.by_id:
                raise ValueError(f"Unknown meal ID {meal_id} on {day_name}")
            meal = catalog.by_id[meal_id]
            if meal_type not in meal.meal_types:
                raise ValueError(f"Meal {meal_id} is not eligible for {meal_type} on {day_name}")
            meals.append(meal)
        all_meals.extend(meals)
        days.append(DailyReport(day_name, meals[0], meals[1], meals[2], _nutrition(meals)))
    total = _nutrition(all_meals)
    average = Nutrition(
        total.calories / DAYS_PER_WEEK,
        total.protein_g / DAYS_PER_WEEK,
        total.fat_g / DAYS_PER_WEEK,
        total.carbs_g / DAYS_PER_WEEK,
    )
    return WeeklyReport(candidate.score, tuple(days), total, average)


def _generation_stats(
    generation: int, phase: str, mutation_rate: Optional[float], population: Sequence[ScoredPlan]
) -> GenerationStats:
    return GenerationStats(
        generation, phase, mutation_rate,
        max(item.score for item in population),
        fmean(item.score for item in population),
        len({item.plan for item in population}),
    )


def run_with_report(
    population: Sequence[ScoredPlan],
    catalog: MealCatalog,
    fitness: Fitness,
    rng: random.Random,
    *,
    generations: int,
    min_exploration_generations: int,
    max_exploration_generations: int,
    stagnation_generations: int,
    improvement_threshold: float,
    exploration_mutation_rate: float,
    exploitation_mutation_rate: float,
    elite_count: int,
    immigrant_count: int,
    tournament_size: int,
    include_statistics: bool = False,
) -> WeeklyReport:
    """Run evolution and report the best plan; optionally retain generation statistics."""
    history: List[GenerationStats] = []
    if include_statistics:
        if not population:
            raise ValueError("Population must not be empty")
        if any(not math.isfinite(candidate.score) for candidate in population):
            raise ValueError("Population scores must be finite")
        history.append(_generation_stats(0, "initial", None, population))

    def observe(generation: int, phase: str, rate: float, current: Tuple[ScoredPlan, ...]) -> None:
        history.append(_generation_stats(generation, phase, rate, current))

    final = run_evolution(
        population, catalog, fitness, rng,
        generations=generations,
        min_exploration_generations=min_exploration_generations,
        max_exploration_generations=max_exploration_generations,
        stagnation_generations=stagnation_generations,
        improvement_threshold=improvement_threshold,
        exploration_mutation_rate=exploration_mutation_rate,
        exploitation_mutation_rate=exploitation_mutation_rate,
        elite_count=elite_count,
        immigrant_count=immigrant_count,
        tournament_size=tournament_size,
        on_generation=observe if include_statistics else None,
    )
    report = weekly_report(max(final, key=lambda item: item.score), catalog)
    if not include_statistics:
        return report
    improvements = []
    best = history[0].best_score
    for entry in history[1:]:
        if entry.best_score > best:
            improvements.append(entry)
            best = entry.best_score
    statistics = RunStatistics(
        population_size=len(population),
        generations=tuple(history),
        improvement_points=tuple(improvements),
        first_exploitation_generation=next(
            entry.generation for entry in history if entry.phase == "exploitation"
        ),
    )
    return WeeklyReport(report.score, report.days, report.weekly_total, report.daily_average, statistics)


def format_report(
    report: WeeklyReport, *, violations: Sequence["Violation"] = (),
    variety: Optional["VarietyViolations"] = None,
) -> str:
    """Format the report as readable plain text."""
    def nutrition(values: Nutrition) -> str:
        return (
            f"{values.calories:.1f} kcal | protein {values.protein_g:.1f} g | "
            f"fat {values.fat_g:.1f} g | carbs {values.carbs_g:.1f} g"
        )

    lines = [f"Weekly meal plan (fitness: {report.score:.3f})"]
    for day in report.days:
        lines.extend((
            f"\n{day.day}",
            f"  Breakfast: {day.breakfast.name}",
            f"  Lunch:     {day.lunch.name}",
            f"  Dinner:    {day.dinner.name}",
            f"  Total: {nutrition(day.total)}",
        ))
    lines.append(f"\nWeekly total: {nutrition(report.weekly_total)}")
    if violations:
        for kind, label in (("minimum", "Weekly shortfalls"),
                            ("maximum", "Weekly excess")):
            parts = []
            for macro, name, unit in (
                ("calories", "calories", "kcal"),
                ("protein_g", "protein", "g"),
                ("fat_g", "fat", "g"),
                ("carbs_g", "carbs", "g"),
            ):
                amount = math.fsum(
                    violation.amount for violation in violations
                    if violation.macro == macro and violation.kind == kind
                )
                parts.append(f"{name} {amount:g} {unit}")
            lines.append(f"{label}: {' | '.join(parts)}")
    if variety is not None and (variety.daily_duplicates or variety.weekly_excess):
        lines.append(
            f"Variety violations: daily duplicate occurrences {variety.daily_duplicates}"
            f" | weekly excess occurrences {variety.weekly_excess}"
        )
    lines.append(f"Daily average: {nutrition(report.daily_average)}")
    if report.statistics is not None:
        stats = report.statistics
        lines.extend((
            "\nRun statistics",
            f"  Population size: {stats.population_size}",
            f"  Generations: {len(stats.generations) - 1}",
            f"  First exploitation generation: {stats.first_exploitation_generation}",
            f"  Initial best: {stats.generations[0].best_score:.3f}",
            f"  Final best: {stats.generations[-1].best_score:.3f}",
            "  Improvements:",
        ))
        if stats.improvement_points:
            lines.extend(
                f"    Generation {entry.generation}: {entry.best_score:.3f}"
                for entry in stats.improvement_points
            )
        else:
            lines.append("    None")
        lines.append("  Generation history (generation | phase | mutation | best | mean | unique plans):")
        for entry in stats.generations:
            rate = "-" if entry.mutation_rate is None else f"{entry.mutation_rate:.3f}"
            lines.append(
                f"    {entry.generation} | {entry.phase} | {rate} | "
                f"{entry.best_score:.3f} | {entry.mean_score:.3f} | {entry.unique_plans}"
            )
    return "\n".join(lines)

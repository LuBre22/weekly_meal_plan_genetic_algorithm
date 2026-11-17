"""Daily nutrition minimums and weekly nutrition maximums."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

from .meals import DAYS_PER_WEEK, MEAL_TYPES, MealCatalog, Plan
from .report import weekly_report
from .selection import ScoredPlan


@dataclass(frozen=True)
class Macros:
    protein_g: float
    fat_g: float
    carbs_g: float


@dataclass(frozen=True)
class Violation:
    period: str
    macro: str
    actual: float
    limit: float
    kind: str

    @property
    def amount(self) -> float:
        return abs(self.actual - self.limit)


@dataclass(frozen=True)
class VarietyViolations:
    daily_duplicates: int
    weekly_excess: int


@dataclass(frozen=True)
class Assessment:
    weekly_protein: float
    violations: Tuple[Violation, ...]
    variety: Optional[VarietyViolations] = None

    @property
    def feasible(self) -> bool:
        return not self.violations and (
            self.variety is None or (
                self.variety.daily_duplicates == 0 and self.variety.weekly_excess == 0
            )
        )

    @property
    def score(self) -> float:
        if self.feasible:
            return self.weekly_protein
        nutrition_penalty = sum(v.amount / max(v.limit, 1.0) for v in self.violations)
        variety_penalty = 0.0
        if self.variety is not None:
            variety_penalty = (
                self.variety.daily_duplicates + self.variety.weekly_excess
            ) / (DAYS_PER_WEEK * len(MEAL_TYPES))
        return -1.0 - nutrition_penalty - variety_penalty


@dataclass(frozen=True)
class MacroConstraints:
    daily_minimum: Macros
    weekly_maximum: Macros
    daily_min_calories: float
    weekly_max_calories: float
    max_meal_occurrences: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_meal_occurrences is not None and (
            isinstance(self.max_meal_occurrences, bool)
            or not isinstance(self.max_meal_occurrences, int)
            or self.max_meal_occurrences < 1
        ):
            raise ValueError("Maximum meal occurrences must be a positive integer")
        limits = [
            (field, getattr(self.daily_minimum, field), getattr(self.weekly_maximum, field))
            for field in ("protein_g", "fat_g", "carbs_g")
        ]
        limits.append(("calories", self.daily_min_calories, self.weekly_max_calories))
        for field, minimum, maximum in limits:
            if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum < 0 or maximum < 0:
                raise ValueError(f"{field} limits must be finite and nonnegative")
            if minimum * DAYS_PER_WEEK > maximum:
                raise ValueError(f"Weekly maximum for {field} is below seven daily minimums")

    def assess(self, plan: Plan, catalog: MealCatalog) -> Assessment:
        report = weekly_report(ScoredPlan(plan, 0), catalog)
        violations = []
        for day in report.days:
            if day.total.calories < self.daily_min_calories:
                violations.append(Violation(
                    day.day, "calories", day.total.calories, self.daily_min_calories, "minimum"
                ))
            for field in ("protein_g", "fat_g", "carbs_g"):
                actual = getattr(day.total, field)
                limit = getattr(self.daily_minimum, field)
                if actual < limit:
                    violations.append(Violation(day.day, field, actual, limit, "minimum"))
        if report.weekly_total.calories > self.weekly_max_calories:
            violations.append(Violation(
                "Week", "calories", report.weekly_total.calories, self.weekly_max_calories, "maximum"
            ))
        for field in ("protein_g", "fat_g", "carbs_g"):
            actual = getattr(report.weekly_total, field)
            limit = getattr(self.weekly_maximum, field)
            if actual > limit:
                violations.append(Violation("Week", field, actual, limit, "maximum"))
        variety = None
        if self.max_meal_occurrences is not None:
            daily_duplicates = sum(len(day) - len(set(day)) for day in plan)
            counts = Counter(meal_id for day in plan for meal_id in day)
            weekly_excess = sum(
                max(0, count - self.max_meal_occurrences) for count in counts.values()
            )
            variety = VarietyViolations(daily_duplicates, weekly_excess)
        return Assessment(report.weekly_total.protein_g, tuple(violations), variety)

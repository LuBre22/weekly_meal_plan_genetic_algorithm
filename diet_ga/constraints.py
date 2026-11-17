"""Daily nutrition minimums and weekly nutrition maximums."""

import math
from dataclasses import dataclass
from typing import Tuple

from .meals import DAYS_PER_WEEK, MealCatalog, Plan
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
class Assessment:
    weekly_protein: float
    violations: Tuple[Violation, ...]

    @property
    def feasible(self) -> bool:
        return not self.violations

    @property
    def score(self) -> float:
        if self.feasible:
            return self.weekly_protein
        return -1.0 - sum(v.amount / max(v.limit, 1.0) for v in self.violations)


@dataclass(frozen=True)
class MacroConstraints:
    daily_minimum: Macros
    weekly_maximum: Macros
    daily_min_calories: float
    weekly_max_calories: float

    def __post_init__(self) -> None:
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
        return Assessment(report.weekly_total.protein_g, tuple(violations))

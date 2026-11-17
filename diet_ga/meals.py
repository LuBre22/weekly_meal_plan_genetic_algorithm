"""Meal catalogue and immutable weekly plan types."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple

MEAL_TYPES = ("breakfast", "lunch", "dinner")
DAYS_PER_WEEK = 7
Day = Tuple[int, int, int]
Plan = Tuple[Day, ...]


@dataclass(frozen=True)
class Meal:
    id: int
    name: str
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    meal_types: Tuple[str, ...]


class MealCatalog:
    def __init__(self, meals: Tuple[Meal, ...]):
        by_id = {}
        eligible = {meal_type: [] for meal_type in MEAL_TYPES}
        for meal in meals:
            if meal.id in by_id:
                raise ValueError(f"Duplicate meal ID: {meal.id}")
            if not meal.name.strip() or not meal.meal_types:
                raise ValueError(f"Meal {meal.id} needs a name and at least one meal type")
            if len(set(meal.meal_types)) != len(meal.meal_types):
                raise ValueError(f"Meal {meal.id} has duplicate meal types")
            if any(not math.isfinite(value) or value < 0 for value in (
                meal.calories, meal.protein_g, meal.fat_g, meal.carbs_g
            )):
                raise ValueError(f"Meal {meal.id} has invalid nutrition values")
            by_id[meal.id] = meal
            for meal_type in meal.meal_types:
                if meal_type not in eligible:
                    raise ValueError(f"Unknown meal type {meal_type!r} for meal {meal.id}")
                eligible[meal_type].append(meal.id)
        for meal_type, ids in eligible.items():
            if not ids:
                raise ValueError(f"No meals eligible for {meal_type}")
        self.by_id: Mapping[int, Meal] = MappingProxyType(by_id)
        self.eligible: Mapping[str, Tuple[int, ...]] = MappingProxyType(
            {meal_type: tuple(ids) for meal_type, ids in eligible.items()}
        )
        self.eligible_index: Mapping[str, Mapping[int, int]] = MappingProxyType(
            {
                meal_type: MappingProxyType({meal_id: index for index, meal_id in enumerate(ids)})
                for meal_type, ids in eligible.items()
            }
        )

    @classmethod
    def from_csv(cls, path: Path) -> "MealCatalog":
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"id", "name", "calories", "protein_g", "fat_g", "carbs_g", "meal_types"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"CSV must include columns: {', '.join(sorted(required))}")
            meals = []
            for line, row in enumerate(reader, start=2):
                try:
                    meals.append(Meal(
                        id=int(row["id"]),
                        name=row["name"].strip(),
                        calories=float(row["calories"]),
                        protein_g=float(row["protein_g"]),
                        fat_g=float(row["fat_g"]),
                        carbs_g=float(row["carbs_g"]),
                        meal_types=tuple(part.strip() for part in row["meal_types"].split("|")),
                    ))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid meal CSV row {line}: {exc}") from exc
        return cls(tuple(meals))

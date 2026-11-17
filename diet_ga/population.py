"""Random generation of weekly plans."""

import random
from typing import List

from .meals import DAYS_PER_WEEK, Day, MEAL_TYPES, MealCatalog, Plan


def random_plan(catalog: MealCatalog, rng: random.Random) -> Plan:
    days = []
    for _ in range(DAYS_PER_WEEK):
        day: Day = (
            rng.choice(catalog.eligible[MEAL_TYPES[0]]),
            rng.choice(catalog.eligible[MEAL_TYPES[1]]),
            rng.choice(catalog.eligible[MEAL_TYPES[2]]),
        )
        days.append(day)
    return tuple(days)


def initialize_population(n: int, catalog: MealCatalog, rng: random.Random) -> List[Plan]:
    if n < 1:
        raise ValueError("Population size must be at least 1")
    return [random_plan(catalog, rng) for _ in range(n)]

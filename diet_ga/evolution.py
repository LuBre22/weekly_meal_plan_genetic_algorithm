"""Fixed-size generational replacement with elites and random immigrants."""

import math
import random
from typing import List, Sequence

from .crossover import crossover
from .meals import MealCatalog
from .mutation import mutate
from .population import random_plan
from .selection import Fitness, ScoredPlan, evaluate, tournament_select


def next_generation(
    population: Sequence[ScoredPlan],
    catalog: MealCatalog,
    fitness: Fitness,
    rng: random.Random,
    *,
    elite_count: int,
    immigrant_count: int,
    tournament_size: int,
    mutation_rate: float,
) -> List[ScoredPlan]:
    n = len(population)
    if n < 1:
        raise ValueError("Population must not be empty")
    if any(not math.isfinite(candidate.score) for candidate in population):
        raise ValueError("Population scores must be finite")
    if isinstance(elite_count, bool) or not isinstance(elite_count, int) or elite_count < 1:
        raise ValueError("Elite count must be a positive integer")
    if immigrant_count < 0 or elite_count + immigrant_count > n:
        raise ValueError("Immigrant count must be nonnegative and elite plus immigrant counts must not exceed the population size")
    if tournament_size < 1 or tournament_size > n:
        raise ValueError("Tournament size must be between 1 and the population size")
    # Validate even when no children are required.
    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1")

    result = sorted(population, key=lambda candidate: candidate.score, reverse=True)[:elite_count]
    children_count = n - elite_count - immigrant_count
    for _ in range(children_count):
        first = tournament_select(population, tournament_size, rng)
        second = tournament_select(population, tournament_size, rng)
        child = crossover(first.plan, second.plan, rng)
        child = mutate(child, catalog, mutation_rate, rng)
        result.append(evaluate(child, fitness))
    result.extend(evaluate(random_plan(catalog, rng), fitness) for _ in range(immigrant_count))
    return result

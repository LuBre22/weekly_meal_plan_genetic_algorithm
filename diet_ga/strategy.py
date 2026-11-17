"""Adaptive exploration and exploitation schedule."""

import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

from .evolution import next_generation
from .meals import MealCatalog
from .selection import Fitness, ScoredPlan

GenerationObserver = Callable[[int, str, float, Tuple[ScoredPlan, ...]], None]


def run_evolution(
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
    on_generation: Optional[GenerationObserver] = None,
) -> List[ScoredPlan]:
    """Run generations, switching from exploration after stagnation or the upper bound."""
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 1:
        raise ValueError("Generations must be a positive integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (min_exploration_generations, max_exploration_generations, stagnation_generations)
    ):
        raise ValueError("Exploration bounds and stagnation generations must be integers")
    if not 0 <= min_exploration_generations <= max_exploration_generations < generations:
        raise ValueError("Exploration bounds must satisfy 0 <= min <= max < generations")
    if stagnation_generations < 1:
        raise ValueError("Stagnation generations must be positive")
    if not math.isfinite(improvement_threshold) or improvement_threshold < 0:
        raise ValueError("Improvement threshold must be finite and nonnegative")
    if (
        not math.isfinite(exploration_mutation_rate)
        or not math.isfinite(exploitation_mutation_rate)
        or not 0 <= exploitation_mutation_rate <= exploration_mutation_rate <= 1
    ):
        raise ValueError("Mutation rates must satisfy 0 <= exploitation <= exploration <= 1")

    current = list(population)
    if not current:
        raise ValueError("Population must not be empty")
    if any(not math.isfinite(candidate.score) for candidate in current):
        raise ValueError("Population scores must be finite")
    baseline = max(candidate.score for candidate in current)
    stagnant = 0
    exploiting = max_exploration_generations == 0
    for generation in range(generations):
        phase = "exploitation" if exploiting else "exploration"
        mutation_rate = exploitation_mutation_rate if exploiting else exploration_mutation_rate
        current = next_generation(
            current, catalog, fitness, rng,
            elite_count=elite_count,
            immigrant_count=immigrant_count,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
        )
        if on_generation is not None:
            on_generation(generation + 1, phase, mutation_rate, tuple(current))
        if not exploiting:
            best = max(candidate.score for candidate in current)
            improvement = best - baseline
            if improvement > 0 and improvement >= improvement_threshold:
                baseline = best
                stagnant = 0
            else:
                stagnant += 1
            completed = generation + 1
            if completed >= max_exploration_generations or (
                completed >= min_exploration_generations and stagnant >= stagnation_generations
            ):
                exploiting = True
    return current

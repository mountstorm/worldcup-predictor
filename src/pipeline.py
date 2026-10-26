"""The prediction pipeline, shared by the CLI and the JSON exporter."""

# Keeps `int | None` and friends working on Python 3.9, where the union
# syntax is not valid in a runtime-evaluated annotation.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import load_matches
from src.elo import compute_elo
from src.events import card_event_report, player_event_report
from src.goal_model import build_training_table, expected_goals, fit_goal_model, score_matrix
from src.simulate import (
    market_probabilities,
    outcome_probabilities,
    score_predictions,
    simulate_final,
    simulate_scores,
)


@dataclass
class Prediction:
    """Everything the pipeline produces for one fixture.

    Attributes:
        teams: [home, away] team names.
        matches: The match table the model was fitted on.
        ratings: Final Elo rating per team.
        history: Per-match Elo history, used for the rating chart.
        lambdas: Expected goals in 90 minutes, per team.
        matrix: Joint scoreline probabilities, home team on rows.
        ninety: Win/draw/win probabilities at 90 minutes.
        sim: Trophy, extra-time and penalty shares from the simulation.
        predictions: Most likely scorelines at 90 and after extra time.
        markets: Goal, clean-sheet and first-scorer probabilities.
        card_events: Card and penalty-awarded probabilities.
        player_events: Per-player goal, assist and penalty probabilities.
    """

    teams: list[str]
    matches: pd.DataFrame
    ratings: dict[str, float]
    history: pd.DataFrame
    lambdas: dict[str, float]
    matrix: np.ndarray
    ninety: dict[str, float]
    sim: dict[str, float]
    predictions: dict[str, list]
    markets: dict[str, float]
    card_events: dict[str, float]
    player_events: dict[str, float]

    @property
    def match_events(self) -> dict[str, float]:
        """Markets and card events together, as the CLI report prints them."""
        return self.markets | self.card_events


def run(cfg: dict, top_n: int | None = None) -> Prediction:
    """Run the full pipeline for the fixture in the config.

    Args:
        cfg: Parsed config.yaml.
        top_n: How many scorelines to rank per stage. Defaults to the
            value in config.yaml.

    Returns:
        A Prediction holding every intermediate the callers need.
    """
    home, away = cfg["match"]["home"], cfg["match"]["away"]
    teams = [home, away]

    matches = load_matches(cfg["data"]["results_csv"], cfg["data"]["min_year"])
    ratings, history = compute_elo(matches, cfg["elo"])

    training = build_training_table(matches, history)
    model = fit_goal_model(training, cfg["goal_model"]["decay_half_life_years"])

    elo_diff = ratings[home] - ratings[away]
    lambdas = {
        home: expected_goals(model, elo_diff),
        away: expected_goals(model, -elo_diff),
    }
    matrix = score_matrix(lambdas[home], lambdas[away], cfg["goal_model"]["max_goals"])
    ninety = outcome_probabilities(matrix)
    markets = market_probabilities(matrix, lambdas[home], lambdas[away], teams)

    scores = simulate_scores(
        lambdas[home],
        lambdas[away],
        cfg["simulation"]["n_runs"],
        cfg["simulation"]["seed"],
        cfg["simulation"]["extra_time_fraction"],
    )
    sim = simulate_final(scores, ratings[home], ratings[away], cfg["elo"]["shootout_scale"])
    predictions = score_predictions(
        scores, top_n if top_n is not None else cfg["score_predictions"]["top_n"]
    )

    return Prediction(
        teams=teams,
        matches=matches,
        ratings=ratings,
        history=history,
        lambdas=lambdas,
        matrix=matrix,
        ninety=ninety,
        sim=sim,
        predictions=predictions,
        markets=markets,
        card_events=card_event_report(cfg["events"]),
        player_events=player_event_report(cfg["events"]),
    )

"""Poisson goal model driven by Elo rating difference."""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor


def _elo_before_match(history: pd.DataFrame) -> pd.DataFrame:
    """Return each team's rating just before each of its matches."""
    shifted = history.copy()
    shifted["prev_rating"] = shifted.groupby("team")["rating"].shift(1)
    return shifted


def _time_decay_weights(dates: pd.Series, half_life_years: float) -> np.ndarray:
    """Return sample weights that halve every half_life_years."""
    age_years = (dates.max() - dates).dt.days / 365.25
    return np.power(0.5, age_years / half_life_years)


def build_training_table(matches: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build one row per team-per-match: goals scored and pre-match Elo diff.

    Args:
        matches: Match results sorted by date.
        history: Post-match Elo snapshots from compute_elo.

    Returns:
        DataFrame with columns date, elo_diff, goals.
    """
    shifted = _elo_before_match(history)
    pre = shifted.dropna(subset=["prev_rating"])
    pre = pre.drop_duplicates(subset=["date", "team"], keep="first")
    pre = pre[["date", "team", "prev_rating"]]

    sides = pd.concat([
        matches.rename(columns={"home_team": "team", "away_team": "opponent",
                                "home_score": "goals"}),
        matches.rename(columns={"away_team": "team", "home_team": "opponent",
                                "away_score": "goals"}),
    ])[["date", "team", "opponent", "goals"]]
    sides = sides.merge(pre, on=["date", "team"])
    sides = sides.merge(
        pre.rename(columns={"team": "opponent", "prev_rating": "opp_rating"}),
        on=["date", "opponent"],
    )
    sides["elo_diff"] = sides["prev_rating"] - sides["opp_rating"]
    return sides[["date", "elo_diff", "goals"]]


def fit_goal_model(training: pd.DataFrame, half_life_years: float) -> PoissonRegressor:
    """Fit expected goals as a Poisson regression on Elo difference.

    Args:
        training: Output of build_training_table.
        half_life_years: Recency-weighting half life.

    Returns:
        Fitted PoissonRegressor.
    """
    weights = _time_decay_weights(training["date"], half_life_years)
    model = PoissonRegressor(alpha=0)
    model.fit(training[["elo_diff"]], training["goals"], sample_weight=weights)
    return model


def expected_goals(model: PoissonRegressor, elo_diff: float) -> float:
    """Return expected goals for a team with the given Elo edge."""
    return float(model.predict(pd.DataFrame({"elo_diff": [elo_diff]}))[0])


def score_matrix(lambda_home: float, lambda_away: float, max_goals: int) -> np.ndarray:
    """Return the joint probability of each scoreline up to max_goals.

    Args:
        lambda_home: Expected goals for the home team.
        lambda_away: Expected goals for the away team.
        max_goals: Largest goal count per team to include.

    Returns:
        (max_goals+1, max_goals+1) array; [i, j] = P(home i, away j).
    """
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    return np.outer(p_home, p_away)
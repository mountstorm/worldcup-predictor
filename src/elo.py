"""Elo rating computation over the full match history."""

import pandas as pd


def expected_score(rating_a: float, rating_b: float, scale: float = 400) -> float:
    """Return the expected score of team A against team B.

    Args:
        rating_a: Team A's Elo rating.
        rating_b: Team B's Elo rating.
        scale: Logistic divisor; larger values dampen the rating edge.

    Returns:
        Expected score in [0, 1].
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / scale))


def _actual_score(home_goals: int, away_goals: int) -> float:
    """Return the home team's actual score: 1 win, 0.5 draw, 0 loss."""
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _match_k(k_factor: float, tournament: str, weights: dict[str, float]) -> float:
    """Return the K factor for a match, weighted by tournament."""
    for name, weight in weights.items():
        if name in tournament:
            return k_factor * weight
    return k_factor


def compute_elo(matches: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Run Elo updates over every match in chronological order.

    Args:
        matches: Match results sorted by date.
        cfg: The `elo` section of the config.

    Returns:
        A (ratings, history) pair: final rating per team, and a long
        DataFrame of (date, team, rating) snapshots after each match.
    """
    ratings: dict[str, float] = {}
    rows = []
    for row in matches.itertuples():
        home = ratings.setdefault(row.home_team, cfg["initial_rating"])
        away = ratings.setdefault(row.away_team, cfg["initial_rating"])
        home_adv = 0 if row.neutral else cfg["home_advantage"]
        expected = expected_score(home + home_adv, away)
        actual = _actual_score(row.home_score, row.away_score)
        k = _match_k(cfg["k_factor"], row.tournament, cfg["tournament_weights"])
        delta = k * (actual - expected)
        ratings[row.home_team] = home + delta
        ratings[row.away_team] = away - delta
        rows.append((row.date, row.home_team, ratings[row.home_team]))
        rows.append((row.date, row.away_team, ratings[row.away_team]))
    history = pd.DataFrame(rows, columns=["date", "team", "rating"])
    return ratings, history
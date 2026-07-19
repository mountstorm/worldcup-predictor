"""Monte Carlo simulation of a single knockout final."""

from collections import Counter

import numpy as np

from src.elo import expected_score


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Split the scoreline matrix into 90-minute win/draw/win probabilities.

    Args:
        matrix: Joint scoreline probabilities from score_matrix.

    Returns:
        Dict with keys home_win, draw, away_win.
    """
    return {
        "home_win": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away_win": float(np.triu(matrix, 1).sum()),
    }


def market_probabilities(
    matrix: np.ndarray, lambda_home: float, lambda_away: float, teams: list[str]
) -> dict[str, float]:
    """Derive match-market probabilities from the scoreline matrix.

    Args:
        matrix: Joint scoreline probabilities, home team on rows.
        lambda_home: Expected goals for the home team.
        lambda_away: Expected goals for the away team.
        teams: [home, away] team names for the labels.

    Returns:
        Dict of display label -> probability (goals, clean sheets,
        first scorer as a goal-rate race).
    """
    home, away = teams
    totals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    p_any_goal = 1.0 - matrix[0, 0]
    home_first_share = lambda_home / (lambda_home + lambda_away)
    return {
        "Both teams score": float(matrix[1:, 1:].sum()),
        "Over 2.5 goals": float(matrix[totals > 2.5].sum()),
        "Under 1.5 goals": float(matrix[totals < 1.5].sum()),
        f"{home} scores first": p_any_goal * home_first_share,
        f"{away} scores first": p_any_goal * (1 - home_first_share),
        f"{home} clean sheet": float(matrix[:, 0].sum()),
        f"{away} clean sheet": float(matrix[0, :].sum()),
    }


def simulate_scores(
    lambda_home: float, lambda_away: float, n_runs: int, seed: int, extra_time_fraction: float
) -> dict:
    """Sample 90-minute and extra-time goals for n_runs finals.

    Args:
        lambda_home: Expected goals for the home team in 90 minutes.
        lambda_away: Expected goals for the away team in 90 minutes.
        n_runs: Number of simulated finals.
        seed: RNG seed for reproducibility.
        extra_time_fraction: Extra time's length as a share of a match.

    Returns:
        Dict of goal arrays: home_90, away_90, home_et, away_et.
    """
    rng = np.random.default_rng(seed)
    return {
        "home_90": rng.poisson(lambda_home, n_runs),
        "away_90": rng.poisson(lambda_away, n_runs),
        "home_et": rng.poisson(lambda_home * extra_time_fraction, n_runs),
        "away_et": rng.poisson(lambda_away * extra_time_fraction, n_runs),
        "pens_draw": rng.random(n_runs),
    }


def simulate_final(
    scores: dict, elo_home: float, elo_away: float, shootout_scale: float
) -> dict[str, float]:
    """Resolve each simulated final through extra time and penalties.

    Args:
        scores: Sampled goal arrays from simulate_scores.
        elo_home: Home team Elo rating.
        elo_away: Away team Elo rating.
        shootout_scale: Elo scale for pens; large values mean mostly luck.

    Returns:
        Dict with keys home_lift, away_lift, extra_time, went_to_pens.
    """
    home_wins = scores["home_90"] > scores["away_90"]
    away_wins = scores["away_90"] > scores["home_90"]
    drawn = ~home_wins & ~away_wins

    full_home = scores["home_90"] + scores["home_et"]
    full_away = scores["away_90"] + scores["away_et"]
    home_wins |= drawn & (full_home > full_away)
    away_wins |= drawn & (full_away > full_home)
    to_pens = ~home_wins & ~away_wins

    pens_home = scores["pens_draw"] < expected_score(elo_home, elo_away, shootout_scale)
    home_wins |= to_pens & pens_home
    away_wins |= to_pens & ~pens_home

    return {
        "home_lift": float(home_wins.mean()),
        "away_lift": float(away_wins.mean()),
        "extra_time": float(drawn.mean()),
        "went_to_pens": float(to_pens.mean()),
    }


def _top_scorelines(home: np.ndarray, away: np.ndarray, top_n: int) -> list[tuple[str, float]]:
    """Return the top_n most frequent scorelines with their shares."""
    counts = Counter(zip(home.tolist(), away.tolist()))
    total = len(home)
    return [(f"{h}-{a}", n / total) for (h, a), n in counts.most_common(top_n)]


def score_predictions(scores: dict, top_n: int) -> dict[str, list]:
    """Rank the most likely scorelines at 90 minutes and after extra time.

    Args:
        scores: Sampled goal arrays from simulate_scores.
        top_n: How many scorelines to keep per stage.

    Returns:
        Dict with keys at_90 and after_et; after_et covers only the
        finals that were level at 90 minutes.
    """
    drawn = scores["home_90"] == scores["away_90"]
    return {
        "at_90": _top_scorelines(scores["home_90"], scores["away_90"], top_n),
        "after_et": _top_scorelines(
            (scores["home_90"] + scores["home_et"])[drawn],
            (scores["away_90"] + scores["away_et"])[drawn],
            top_n,
        ),
    }

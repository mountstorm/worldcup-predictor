"""Match-event probabilities (goals, assists, penalties, cards) from base rates."""

import math

from scipy.stats import poisson


def _at_least_one(rate: float) -> float:
    """Return P(count >= 1) for a Poisson event with the given rate."""
    return 1.0 - math.exp(-rate)


def player_scores_probability(player: dict) -> float:
    """Return P(the player scores), including whether he plays."""
    return player["plays_probability"] * _at_least_one(player["goals_per_match"])


def player_assists_probability(player: dict) -> float:
    """Return P(the player records an assist), including whether he plays."""
    return player["plays_probability"] * _at_least_one(player["assists_per_match"])


def player_penalty_goal_probability(player: dict, penalty_awarded_rate: float) -> float:
    """Return P(the player scores a penalty in the match).

    Chain: his team is awarded a pen (half the per-match rate), he is
    on the pitch and takes it, and he converts.
    """
    team_pen_rate = penalty_awarded_rate / 2
    return (
        _at_least_one(team_pen_rate)
        * player["plays_probability"]
        * player["penalty_take_share"]
        * player["penalty_conversion"]
    )


def yellow_card_distribution(cfg: dict) -> dict[str, float]:
    """Return expected yellows and P(total goes over 4.5)."""
    rate = cfg["world_cup_match"]["yellow_cards_per_match"]
    return {
        "expected": rate,
        "over_4_5": float(1 - poisson.cdf(4, rate)),
    }


def red_card_probability(cfg: dict) -> float:
    """Return P(at least one red card in the final)."""
    return _at_least_one(cfg["world_cup_match"]["red_card_rate"])


def player_event_report(cfg: dict) -> dict[str, float]:
    """Return goal/assist/penalty probabilities for every configured player."""
    pen_rate = cfg["world_cup_match"]["penalty_awarded_rate"]
    report = {}
    for player in cfg["players"]:
        label = f"{player['name']} ({player['team']})"
        report[f"{label} scores"] = player_scores_probability(player)
        report[f"{label} assists"] = player_assists_probability(player)
        if player["penalty_take_share"] >= cfg["penalty_display_min_share"]:
            report[f"{label} scores a penalty"] = player_penalty_goal_probability(player, pen_rate)
    return dict(sorted(report.items(), key=lambda kv: kv[1], reverse=True))


def card_event_report(cfg: dict) -> dict[str, float]:
    """Return card and penalty-awarded probabilities for the match."""
    yellows = yellow_card_distribution(cfg)
    return {
        "A penalty is awarded": _at_least_one(cfg["world_cup_match"]["penalty_awarded_rate"]),
        f"Over 4.5 yellow cards (exp. {yellows['expected']:.1f})": yellows["over_4_5"],
        "At least one red card": red_card_probability(cfg),
    }
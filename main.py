"""Predict the 2026 World Cup final from historical match data."""

from pathlib import Path

import yaml

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
from src.visualize import plot_elo_history, plot_events, plot_outcomes, plot_score_heatmap


def load_config(path: str) -> dict:
    """Load the YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def print_report(teams: list[str], elos: dict, lambdas: dict, ninety: dict, sim: dict) -> None:
    """Print the prediction summary to stdout."""
    home, away = teams
    print(f"\n=== {home} vs {away} ===")
    print(f"Elo: {home} {elos[home]:.0f} | {away} {elos[away]:.0f}")
    print(f"Expected goals: {home} {lambdas[home]:.2f} | {away} {lambdas[away]:.2f}")
    print(f"90 minutes: {home} win {ninety['home_win']:.1%} | "
          f"draw {ninety['draw']:.1%} | {away} win {ninety['away_win']:.1%}")
    print(f"Extra time in {sim['extra_time']:.1%} of finals, pens in {sim['went_to_pens']:.1%}")
    print(f"Trophy: {home} {sim['home_lift']:.1%} | {away} {sim['away_lift']:.1%}")


def print_scores(predictions: dict, teams: list[str]) -> None:
    """Print the most likely scorelines at each stage."""
    home, away = teams
    print(f"\n=== Most likely scores ({home}-{away}) ===")
    for stage, label in (("at_90", "At 90 minutes"), ("after_et", "After extra time (if level at 90)")):
        ranked = ", ".join(f"{score} ({share:.1%})" for score, share in predictions[stage])
        print(f"{label}: {ranked}")


def print_events(title: str, events: dict[str, float]) -> None:
    """Print each event probability under a section title."""
    print(f"\n=== {title} ===")
    for label, prob in events.items():
        print(f"{label}: {prob:.1%}")


def main() -> None:
    """Run the full pipeline: data, Elo, goal model, simulation, charts."""
    cfg = load_config("config.yaml")
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
        lambdas[home], lambdas[away], cfg["simulation"]["n_runs"], cfg["simulation"]["seed"],
        cfg["simulation"]["extra_time_fraction"],
    )
    sim = simulate_final(scores, ratings[home], ratings[away], cfg["elo"]["shootout_scale"])
    predictions = score_predictions(scores, cfg["score_predictions"]["top_n"])

    match_events = markets | card_event_report(cfg["events"])
    player_events = player_event_report(cfg["events"])

    out_dir = cfg["output"]["dir"]
    Path(out_dir).mkdir(exist_ok=True)
    for path in (
        plot_elo_history(history, teams, cfg["viz"], out_dir),
        plot_score_heatmap(matrix, teams, cfg["viz"], out_dir),
        plot_outcomes(sim, ninety, teams, cfg["viz"], out_dir),
        plot_events(match_events, "Match & team event probabilities", "match_events.png",
                    cfg["viz"], out_dir),
        plot_events(player_events, "Player event probabilities", "player_events.png",
                    cfg["viz"], out_dir),
    ):
        print(f"chart: {path}")

    print_report(teams, ratings, lambdas, ninety, sim)
    print_scores(predictions, teams)
    print_events("Match & team events", match_events)
    print_events("Player events", player_events)


if __name__ == "__main__":
    main()

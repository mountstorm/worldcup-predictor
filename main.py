"""Predict the 2026 World Cup final from historical match data."""

from pathlib import Path

import yaml

from src.pipeline import Prediction, run
from src.visualize import (
    plot_elo_history,
    plot_events,
    plot_outcomes,
    plot_score_heatmap,
)


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


def write_charts(p: Prediction, cfg: dict) -> None:
    """Render every chart to the output directory."""
    out_dir = cfg["output"]["dir"]
    Path(out_dir).mkdir(exist_ok=True)
    for path in (
        plot_elo_history(p.history, p.teams, cfg["viz"], out_dir),
        plot_score_heatmap(p.matrix, p.teams, cfg["viz"], out_dir),
        plot_outcomes(p.sim, p.ninety, p.teams, cfg["viz"], out_dir),
        plot_events(p.match_events, "Match & team event probabilities", "match_events.png",
                    cfg["viz"], out_dir),
        plot_events(p.player_events, "Player event probabilities", "player_events.png",
                    cfg["viz"], out_dir),
    ):
        print(f"chart: {path}")


def main() -> None:
    """Run the full pipeline: data, Elo, goal model, simulation, charts."""
    cfg = load_config("config.yaml")
    p = run(cfg)

    write_charts(p, cfg)

    print_report(p.teams, p.ratings, p.lambdas, p.ninety, p.sim)
    print_scores(p.predictions, p.teams)
    print_events("Match & team events", p.match_events)
    print_events("Player events", p.player_events)


if __name__ == "__main__":
    main()

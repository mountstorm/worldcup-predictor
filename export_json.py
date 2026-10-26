"""Dump the prediction as JSON, for consumption by a web dashboard.

Writes every number the dashboard at muzaffarkhaydarov.com/projects/worldcup
renders, so the page never hardcodes a figure the model did not produce.

    python3 export_json.py                  # -> output/prediction.json
    python3 export_json.py --stdout         # print instead of writing
    python3 export_json.py -o path/to.json  # write somewhere else
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.pipeline import Prediction, run

SCORE_TOP_N = 5


def elo_history_by_year(p: Prediction) -> dict[str, list[list[float]]]:
    """Return each finalist's year-end Elo, for the rating chart.

    Args:
        p: A completed Prediction.

    Returns:
        Dict of team -> [[year, rating], ...], one point per year.
    """
    out: dict[str, list[list[float]]] = {}
    for team in p.teams:
        rows = p.history[p.history["team"] == team][["date", "rating"]].copy()
        rows["year"] = rows["date"].dt.year
        yearly = rows.groupby("year")["rating"].last()
        out[team] = [[int(y), round(float(r), 1)] for y, r in yearly.items()]
    return out


def build_payload(p: Prediction, cfg: dict) -> dict:
    """Assemble the JSON payload from a completed Prediction.

    Args:
        p: A completed Prediction.
        cfg: Parsed config.yaml, for the tunables worth publishing.

    Returns:
        A JSON-serialisable dict.
    """
    home, away = p.teams
    max_goals = cfg["viz"]["heatmap_max_goals"]

    return {
        "match": {"home": home, "away": away},
        "dataset": {
            "matches": int(len(p.matches)),
            "min_year": cfg["data"]["min_year"],
            "runs": cfg["simulation"]["n_runs"],
            "seed": cfg["simulation"]["seed"],
        },
        "elo": {t: round(float(p.ratings[t]), 1) for t in p.teams},
        "expected_goals": {t: round(float(p.lambdas[t]), 3) for t in p.teams},
        "ninety": {k: round(float(v), 5) for k, v in p.ninety.items()},
        "simulation": {k: round(float(v), 5) for k, v in p.sim.items()},
        "scores": {
            stage: [[score, round(float(share), 5)] for score, share in ranked]
            for stage, ranked in p.predictions.items()
        },
        "markets": {k: round(float(v), 5) for k, v in p.markets.items()},
        "match_events": {k: round(float(v), 5) for k, v in p.card_events.items()},
        "player_events": {k: round(float(v), 5) for k, v in p.player_events.items()},
        "heatmap": {
            "max_goals": max_goals,
            "matrix": [
                [float(p.matrix[i][j]) for j in range(max_goals + 1)]
                for i in range(max_goals + 1)
            ],
        },
        "elo_history": elo_history_by_year(p),
        "config": {
            "k_factor": cfg["elo"]["k_factor"],
            "home_advantage": cfg["elo"]["home_advantage"],
            "shootout_scale": cfg["elo"]["shootout_scale"],
            "decay_half_life_years": cfg["goal_model"]["decay_half_life_years"],
            "extra_time_fraction": cfg["simulation"]["extra_time_fraction"],
        },
    }


def main() -> None:
    """Run the pipeline and emit the payload as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--out", default=None, help="output path")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    payload = build_payload(run(cfg, top_n=SCORE_TOP_N), cfg)
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.stdout:
        print(text)
        return

    out = Path(args.out or Path(cfg["output"]["dir"]) / "prediction.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"wrote {out} ({len(text):,} bytes)")


if __name__ == "__main__":
    main()

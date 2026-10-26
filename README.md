# worldcup-predictor

Predicts a projected 2026 World Cup final (Argentina vs Spain) from 32,401
international matches since 1990: Elo ratings feed a recency-weighted Poisson
goal model, resolved over 10,000 Monte Carlo finals.

The two teams come out within Monte Carlo noise of each other — the model says
coin flip, not favourite.

**Live dashboard:** [muzaffarkhaydarov.com/projects/worldcup](https://muzaffarkhaydarov.com/projects/worldcup)

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scipy scikit-learn matplotlib pyyaml
.venv/bin/python main.py
```

Prints win/score/event probabilities and writes charts to `output/`. All
tunables live in `config.yaml`.

## Export for the dashboard

```bash
.venv/bin/python export_json.py          # -> output/prediction.json
.venv/bin/python export_json.py --stdout # print instead of writing
```

`export_json.py` runs the same pipeline as `main.py` and emits every number the
web dashboard renders, so the page never hardcodes a figure the model did not
produce. After changing `config.yaml`, re-run it and copy the JSON into the
site's `src/lib/worldcup.ts`.

Both entry points share `src/pipeline.py`, so the printed report and the JSON
can never drift apart.

## Layout

| Path | What it does |
|---|---|
| `main.py` | CLI: prints the report, writes the charts |
| `export_json.py` | Emits `output/prediction.json` for the dashboard |
| `src/pipeline.py` | The pipeline both entry points call |
| `src/elo.py` | Elo ratings, K-weighted by tournament |
| `src/goal_model.py` | Poisson goal model and scoreline matrix |
| `src/simulate.py` | Monte Carlo resolution through ET and pens |
| `src/events.py` | Card, penalty and player probabilities from base rates |
| `src/visualize.py` | Matplotlib charts |

## Caveats

- The final is a projection; Argentina and Spain reaching it is an assumption,
  not a result of this model.
- Elo knows results, not squads — injuries, form and selection are not modelled.
- Player and card probabilities come from published base rates, not from the
  simulation.
- Monte Carlo noise at 10,000 runs is about ±1pp near 50%, which is wider than
  the gap between the two teams.

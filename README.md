# worldcup-predictor

Predicts the 2026 World Cup final (Argentina vs Spain) from ~35,000 international matches: Elo ratings feed a Poisson goal model, resolved over 10,000 Monte Carlo simulations.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scipy scikit-learn matplotlib pyyaml
.venv/bin/python main.py
```

Prints win/score/event probabilities and writes charts to `output/`. All tunables live in `config.yaml`.
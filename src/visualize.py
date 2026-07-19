"""Chart output for the final prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _new_axes(cfg: dict, figsize: tuple) -> tuple:
    """Return a styled (figure, axes) pair on the standard surface."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=cfg["surface"])
    ax.set_facecolor(cfg["surface"])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(cfg["grid"])
    ax.tick_params(colors=cfg["muted"], labelsize=9)
    return fig, ax


def _save(fig, cfg: dict, out_dir: str, name: str) -> str:
    """Save the figure as a PNG and return its path."""
    path = str(Path(out_dir) / name)
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    return path


def _barh_chart(
    events: dict[str, float], colors: list, title: str, filename: str, cfg: dict, out_dir: str
) -> str:
    """Plot label -> probability pairs as a horizontal bar chart.

    Args:
        events: Label -> probability pairs, drawn top to bottom.
        colors: One bar color per label.
        title: Chart title.
        filename: Output PNG name.
        cfg: The `viz` section of the config.
        out_dir: Directory for the PNG.

    Returns:
        Path of the saved chart.
    """
    labels = list(events)
    values = list(events.values())
    fig, ax = _new_axes(cfg, (8, 0.55 * len(labels) + 0.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.55)
    for yi, value in zip(y, values):
        ax.text(value + 0.008, yi, f"{value:.1%}", va="center", color=cfg["ink"], fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=cfg["ink"], fontsize=10)
    ax.set_xlim(0, max(values) * 1.25)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", color=cfg["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=cfg["ink"], fontsize=13, loc="left")
    return _save(fig, cfg, out_dir, filename)


def plot_elo_history(history: pd.DataFrame, teams: list[str], cfg: dict, out_dir: str) -> str:
    """Plot both teams' Elo ratings over time with direct end labels.

    Args:
        history: Elo snapshots from compute_elo.
        teams: The two team names to plot.
        cfg: The `viz` section of the config.
        out_dir: Directory for the PNG.

    Returns:
        Path of the saved chart.
    """
    fig, ax = _new_axes(cfg, (9, 4.5))
    final = {t: history[history["team"] == t]["rating"].iloc[-1] for t in teams}
    leader = max(teams, key=final.get)
    for team in teams:
        series = history[history["team"] == team]
        color = cfg["team_colors"][team]
        ax.plot(series["date"], series["rating"], color=color, linewidth=2, label=team)
        # Stack the end labels by final rating so they never collide.
        offset = 10 if team == leader else -10
        ax.annotate(
            f"{team}  {final[team]:.0f}",
            (series["date"].iloc[-1], final[team]),
            textcoords="offset points", xytext=(8, offset),
            color=cfg["ink"], fontsize=10, fontweight="bold", va="center",
        )
    ax.grid(axis="y", color=cfg["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=cfg["ink"], fontsize=9, loc="upper left")
    ax.set_title("Elo rating history", color=cfg["ink"], fontsize=13, loc="left")
    ax.margins(x=0.12)
    return _save(fig, cfg, out_dir, "elo_history.png")


def plot_score_heatmap(matrix: np.ndarray, teams: list[str], cfg: dict, out_dir: str) -> str:
    """Plot scoreline probabilities as a sequential heatmap.

    Args:
        matrix: Joint scoreline probabilities, home team on rows.
        teams: [home, away] team names.
        cfg: The `viz` section of the config.
        out_dir: Directory for the PNG.

    Returns:
        Path of the saved chart.
    """
    from matplotlib.colors import LinearSegmentedColormap

    show = min(cfg["heatmap_max_goals"], matrix.shape[0] - 1)
    sub = matrix[: show + 1, : show + 1]
    cmap = LinearSegmentedColormap.from_list("seq", [cfg["surface"]] + cfg["sequential"])

    fig, ax = _new_axes(cfg, (6, 5))
    ax.imshow(sub, cmap=cmap, origin="lower")
    for i in range(show + 1):
        for j in range(show + 1):
            value = sub[i, j]
            ink = "#ffffff" if value > sub.max() * 0.6 else cfg["ink"]
            ax.text(j, i, f"{value:.1%}", ha="center", va="center", color=ink, fontsize=9)
    ax.set_xticks(range(show + 1))
    ax.set_yticks(range(show + 1))
    ax.set_xlabel(f"{teams[1]} goals", color=cfg["muted"])
    ax.set_ylabel(f"{teams[0]} goals", color=cfg["muted"])
    ax.set_title("Scoreline probabilities (90 min)", color=cfg["ink"], fontsize=13, loc="left")
    return _save(fig, cfg, out_dir, "score_heatmap.png")


def plot_outcomes(sim: dict, ninety: dict, teams: list[str], cfg: dict, out_dir: str) -> str:
    """Plot who-lifts-the-trophy and 90-minute outcome probabilities.

    Args:
        sim: Simulation shares from simulate_final.
        ninety: 90-minute probabilities from outcome_probabilities.
        teams: [home, away] team names.
        cfg: The `viz` section of the config.
        out_dir: Directory for the PNG.

    Returns:
        Path of the saved chart.
    """
    home, away = teams
    events = {
        f"{home} lift the trophy": sim["home_lift"],
        f"{away} lift the trophy": sim["away_lift"],
        f"{home} win in 90'": ninety["home_win"],
        "Draw after 90'": ninety["draw"],
        f"{away} win in 90'": ninety["away_win"],
    }
    colors = [
        cfg["team_colors"][home], cfg["team_colors"][away],
        cfg["team_colors"][home], cfg["muted"], cfg["team_colors"][away],
    ]
    return _barh_chart(events, colors, "Final outcome probabilities", "outcomes.png", cfg, out_dir)


def plot_events(events: dict[str, float], title: str, filename: str, cfg: dict, out_dir: str) -> str:
    """Plot event probabilities as a single-hue bar chart.

    Args:
        events: Label -> probability pairs.
        title: Chart title.
        filename: Output PNG name.
        cfg: The `viz` section of the config.
        out_dir: Directory for the PNG.

    Returns:
        Path of the saved chart.
    """
    colors = [cfg["bar_color"]] * len(events)
    return _barh_chart(events, colors, title, filename, cfg, out_dir)
"""Loading and filtering of the international match results dataset."""

import pandas as pd


def load_matches(csv_path: str, min_year: int) -> pd.DataFrame:
    """Load match results and keep games from min_year onward.

    Args:
        csv_path: Path to the results CSV file.
        min_year: Earliest year of matches to keep.

    Returns:
        DataFrame with parsed dates, sorted chronologically.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"].dt.year >= min_year]
    return df.sort_values("date").reset_index(drop=True)

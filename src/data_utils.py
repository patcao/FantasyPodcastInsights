from typing import Optional
import pandas as pd
from pathlib import Path


fantasy_point_weights = {
    "points": 1,
    "reboundsTotal": 1.25,
    "assists": 1.5,
    "steals": 2,
    "blocks": 2,
    "turnovers": -0.5,
    "double-double": 1.5,
    "triple-double": 3,
}


def get_repo_root():
    """Find the root of the repository based on the location of the .git directory."""
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.root:
        if (current_dir / ".git").is_dir():  # Check for .git folder
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError("Repository root not found (no .git directory)")


def load_and_filter(
    file_path: str, time_context: Optional[tuple] = None, time_column: str = "game_date"
):
    """
    Load a CSV file and filter dfs based on a time context.

    Args:
        file_path (str): Path to the CSV file.
        time_context (tuple): A tuple containing start and end time-like objects. Can be strings, datetime, or Pandas Timestamp.
        time_column (str): The column name containing time data.

    Returns:
        pd.DataFrame: Filtered DataFrame based on the time context.
    """
    try:
        df = pd.read_csv(file_path)
        if time_column not in df.columns:
            raise KeyError(f"Time column '{time_column}' not found in the DataFrame.")

        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        df = df.dropna(subset=[time_column])

        if time_context:
            if not isinstance(time_context, tuple) or len(time_context) != 2:
                raise ValueError(
                    "time_context must be a tuple with two elements (start_time, end_time)."
                )

            # Parse the time_context into datetime objects
            start_time, end_time = time_context

            if start_time is not None:
                start_time = pd.to_datetime(start_time, errors="coerce")
                if start_time is pd.NaT:
                    raise ValueError(f"Invalid start_time: {time_context[0]}")
                df = df[df[time_column] >= start_time]

            if end_time is not None:
                end_time = pd.to_datetime(end_time, errors="coerce")
                if end_time is pd.NaT:
                    raise ValueError(f"Invalid end_time: {time_context[1]}")
                df = df[df[time_column] <= end_time]

        return df

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while loading and filtering the CSV: {e}"
        )


def compute_fantasy_points(df):
    # Base fantasy points (excluding bonuses)
    base_points = sum(
        [
            df[col_name] * weight
            for col_name, weight in fantasy_point_weights.items()
            if col_name not in ["double-double", "triple-double"]
        ]
    )

    # Count the number of stats >= 10 for double-double/triple-double logic
    stats = ["points", "reboundsTotal", "assists", "steals", "blocks"]
    double_digit_count = sum(df[stat] >= 10 for stat in stats)

    # Calculate double-double and triple-double bonuses
    double_double_bonus = (double_digit_count >= 2) * fantasy_point_weights[
        "double-double"
    ]
    triple_double_bonus = (double_digit_count >= 3) * fantasy_point_weights[
        "triple-double"
    ]

    df["fantasyPoints"] = base_points + double_double_bonus + triple_double_bonus
    return base_points + double_double_bonus + triple_double_bonus


def preprocess_box_scores(df: pd.DataFrame):
    def convert_to_decimal_minutes(value):
        if pd.isna(value):
            return 0
        try:
            minutes, seconds = map(int, value.split(":"))
            return minutes + seconds / 60
        except ValueError:
            # Handle any unexpected format
            return 0

    df = df.sort_values(["game_date", "gameId", "personId"])

    # Convert minutes from mm::ss to decimal format
    df = df.assign(minutes=df["minutes"].apply(convert_to_decimal_minutes))
    # Compute fantasy points
    df = df.assign(fantasyPoints=compute_fantasy_points(df))

    return df.reset_index(drop=True)



from typing import Optional
import pandas as pd
from pathlib import Path
from src.utils import get_repo_root
from src.constants import DateLike

# These columns are clean and suitable for training models on
CLEAN_COLUMNS = [
    "season_year",
    "game_date",
    "gameId",
    "matchup",
    # "teamId",
    # "teamCity",
    "teamName",
    # "teamTricode",
    "teamSlug",
    "personId",
    "personName",
    # "position",
    # "comment",
    # "jerseyNum",
    "minutes",
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "fieldGoalsPercentage",
    "threePointersMade",
    "threePointersAttempted",
    "threePointersPercentage",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "freeThrowsPercentage",
    "reboundsOffensive",
    "reboundsDefensive",
    "reboundsTotal",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "foulsPersonal",
    "points",
    "plusMinusPoints",
    "fantasyPoints",
    "projectedFantasyPoints",
    "outperformed",
]


def load_clean_scores(seasons: list[str] = None, columns: list[str] = CLEAN_COLUMNS):
    """
    Load cleaned regular season box score data.

    Read a Parquet file containing cleaned regular season box scores,
    selects specified columns, and filters the data by the given seasons.

    Parameters
    ----------
        seasons: optional[list[str]]
            A list of season strings to filter the data. Each season should be in the format "YYYY-YY".
            Defaults to all seasons if not provided.
        columns: optional[list[str]]
            A list of column names to keep from the dataset.
            Defaults to CLEAN_COLUMNS.

    Return
    ------
        pd.DataFrame:
            A filtered DataFrame containing the specified seasons and columns.
    """

    data_path = get_repo_root() / "data/processed/regular_season_box_scores.pq"
    df = pd.read_parquet(data_path, columns=columns)

    if seasons is not None:
        return df[df["season_year"].isin(seasons)]
    else:
        return df


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

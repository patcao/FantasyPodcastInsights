import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.constants import DateLike
from src.utils import get_repo_root

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
    "comment",
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
    "outperform_next",
    "outperform_next_5",
    "outperform_next_10",
    "injured_next",
    "injured",
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
    df["fantasyDiff"] = df["fantasyPoints"] - df["projectedFantasyPoints"]

    df = df.drop(columns="comment")

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


def add_lagged_features(df: pd.DataFrame, feature_names: list[str], max_lag: int):
    assign_map = {}

    for target_col in feature_names:
        for i in range(1, max_lag + 1):
            assign_map[f"{target_col}_lag_{i}"] = df.groupby("personId")[
                target_col
            ].shift(i)

    return df.assign(**assign_map), list(assign_map.keys())


def create_training_dataset(
    scores: pd.DataFrame,
    non_lag_features: str,
    lag_features: str,
    target_col: str,
    diff_threshold: float = 0,
):
    df, lagged_feat_names = add_lagged_features(scores, lag_features, 5)
    df = df.assign(outperformed=np.where(df["fantasyDiff"] > diff_threshold, 1, 0))
    df.dropna(inplace=True)

    return {
        "df": df,
        "non_lag_features": non_lag_features,
        "lagged_feat_names": lagged_feat_names,
        "target_col": target_col,
    }


class PodcastContainer:
    """
    A class for managing a collection of episodes of multiple podcasts.

    Handles operations like file extraction, name normalization,
    and merging metadata from a manifest.
    """

    ROTOWIRE_DIR = get_repo_root() / "data/raw/DG RFB Transcripts/"
    NBATODAY_DIR = get_repo_root() / "data/raw/DG FNT Transcripts/"

    def __init__(self, podcast_directories: Optional[dict[str, str]] = None):
        if podcast_directories is None:
            self.podcast_directories = {
                "rotowire": get_repo_root() / "data/raw/DG RFB Transcripts/",
                "nbatoday": get_repo_root() / "data/raw/DG FNT Transcripts/",
            }
        else:
            self.podcast_directories = {
                name: Path(directory) for name, directory in podcast_directories.items()
            }

    @staticmethod
    def _normalize_string(text: str) -> str:
        """
        Normalize a string by removing punctuation, converting to lowercase,
        and replacing spaces with underscores.

        Parameters
        ----------
        text : str
            The input string to normalize.
        """
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower()
        return re.sub(r"\s+", "_", text.strip())

    @staticmethod
    def extract_name_from_path(file_path: Path) -> str:
        """
        Extract the meaningful name from a given file path.

        Parameters
        ----------
        file_path : Path
            The file path to process.

        Returns
        -------
        str
            The extracted meaningful name.
        """
        filename = file_path.stem
        # Remove the `_transcript` suffix and clean up underscores
        filename = re.sub(r"_transcript.*$", "", filename)
        return filename.replace("_", " ").strip()

    def _list_files(self, directory: Path, suffix: str) -> list[Path]:
        """
        List all files in the directory with a specific suffix.

        Parameters
        ----------
        directory : Path
            The directory to search for files.
        suffix : str
            The file suffix (e.g., 'txt', 'csv') to filter by.

        Returns
        -------
        list[Path]
            A list of Path objects representing the files with the specified suffix.
        """
        if not directory.is_dir():
            raise ValueError(f"The provided path '{directory}' is not a directory.")
        return [file for file in directory.glob(f"*{suffix}") if file.is_file()]

    def _manifest_file(self, directory: Path) -> Path:
        """
        Retrieve the single manifest file (CSV) in the directory.

        Parameters
        ----------
        directory : Path
            The directory to search for the manifest file.

        Returns
        -------
        Path
            The path to the manifest file.

        Raises
        ------
        AssertionError
            If there is not exactly one CSV file in the directory.
        """
        csv_files = self._list_files(directory, "csv")
        assert (
            len(csv_files) == 1
        ), "Expected exactly one manifest file in the directory."
        return csv_files[0]

    def podcast_files(self, podcast_name: str) -> list[Path]:
        """
        List all podcast transcript files in the directory for a given podcast.

        Parameters
        ----------
        podcast_name : str
            The name of the podcast.

        Returns
        -------
        list[Path]
            A list of Path objects representing the podcast files.
        """
        if podcast_name not in self.podcast_directories:
            raise ValueError(f"Podcast '{podcast_name}' not found in the directories.")
        return self._list_files(self.podcast_directories[podcast_name], "txt")

    def manifest_data(self, podcast_name: str) -> pd.DataFrame:
        """
        Load the manifest data from the CSV file for a given podcast.

        Parameters
        ----------
        podcast_name : str
            The name of the podcast.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the manifest data.
        """
        if podcast_name not in self.podcast_directories:
            raise ValueError(f"Podcast '{podcast_name}' not found in the directories.")
        return pd.read_csv(self._manifest_file(self.podcast_directories[podcast_name]))

    def get_all_podcast_names(self) -> list[str]:
        return list(self.podcast_directories.keys())

    def get_all_episodes(self) -> pd.DataFrame:
        return (
            pd.concat(
                [
                    self.get_episodes_for_podcast(name).assign(podcast_name=name)
                    for name in self.get_all_podcast_names()
                ]
            )
            .sort_values("publication_date")
            .reset_index(drop=True)
        )

    def get_episodes_for_date(self, for_date: DateLike) -> pd.DataFrame:
        """
        Get all podcast episodes for a specific date across all podcast directories.

        Parameters
        ----------
        for_date : DateLike
            The date to filter episodes by.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all episodes for the specified date.
        """
        for_date = pd.to_datetime(for_date).date()
        all_episodes = []

        for podcast_name in self.get_all_podcast_names():
            episodes = self.get_episodes_for_podcast(podcast_name)
            episodes_for_date = episodes[episodes["publication_date"] == for_date]
            all_episodes.append(episodes_for_date)

        return pd.concat(all_episodes, ignore_index=True)

    def get_episodes_for_podcast(self, podcast_name: str) -> pd.DataFrame:
        """
        Combine podcast file data with metadata from the manifest for a given podcast.

        Parameters
        ----------
        podcast_name : str
            The name of the podcast.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing combined podcast file and manifest metadata.
            Columns include:
            - file_name
            - file_path
            - publication_date
            - duration
        """

        def read_file_text(file_path: Path) -> str:
            with file_path.open("r", encoding="utf-8") as f:
                return f.read()

        # Create a DataFrame for podcast files
        file_data = pd.DataFrame(
            [
                {
                    "file_path": str(p),
                    "file_name": self._normalize_string(self.extract_name_from_path(p)),
                    "content": read_file_text(p),
                }
                for p in self.podcast_files(podcast_name)
            ]
        )

        # Load and process manifest data
        mf_data = self.manifest_data(podcast_name)
        mf_data["publication_date"] = pd.to_datetime(
            pd.to_datetime(mf_data["publication_date"]).dt.date
        )
        mf_data["title"] = mf_data["title"].apply(self._normalize_string)
        mf_data = mf_data[["title", "publication_date", "duration"]]

        # Merge file data with manifest metadata
        result = (
            pd.merge(
                file_data, mf_data, left_on="file_name", right_on="title", how="left"
            )
            .sort_values(by="publication_date")
            .reset_index(drop=True)
        )

        return result[
            ["publication_date", "file_name", "file_path", "content", "duration"]
        ]

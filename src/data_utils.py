from typing import Optional
import pandas as pd
from pathlib import Path
from src.utils import get_repo_root
from src.constants import DateLike
import re

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


class PodcastContainer:
    """
    A class for managing podcast data, including file extraction, name normalization,
    and merging metadata from a manifest.
    """

    def __init__(self):
        self.directory_path = get_repo_root() / "data/raw/rotowire_2023_2024"

    @staticmethod
    def _normalize_string(text: str) -> str:
        """
        Normalize a string by removing punctuation, converting to lowercase,
        and replacing spaces with underscores.

        Parameters
        ----------
        text : str
            The input string to normalize.

        Returns
        -------
        str
            The normalized string.
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

    def _list_files(self, suffix: str) -> list[Path]:
        """
        List all files in the directory with a specific suffix.

        Parameters
        ----------
        suffix : str
            The file suffix (e.g., 'txt', 'csv') to filter by.

        Returns
        -------
        list[Path]
            A list of Path objects representing the files with the specified suffix.
        """
        if not self.directory_path.is_dir():
            raise ValueError(
                f"The provided path '{self.directory_path}' is not a directory."
            )
        return [
            file for file in self.directory_path.glob(f"*{suffix}") if file.is_file()
        ]

    def _manifest_file(self) -> Path:
        """
        Retrieve the single manifest file (CSV) in the directory.

        Returns
        -------
        Path
            The path to the manifest file.

        Raises
        ------
        AssertionError
            If there is not exactly one CSV file in the directory.
        """
        csv_files = self._list_files("csv")
        assert (
            len(csv_files) == 1
        ), "Expected exactly one manifest file in the directory."
        return csv_files[0]

    def podcast_files(self) -> list[Path]:
        """
        List all podcast transcript files in the directory.

        Returns
        -------
        list[Path]
            A list of Path objects representing the podcast files.
        """
        return self._list_files("txt")

    def manifest_data(self) -> pd.DataFrame:
        """
        Load the manifest data from the CSV file.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the manifest data.
        """
        return pd.read_csv(self._manifest_file())

    def get_podcast_data(self) -> pd.DataFrame:
        """
        Combine podcast file data with metadata from the manifest.

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
                for p in self.podcast_files()
            ]
        )

        # Load and process manifest data
        mf_data = self.manifest_data()
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

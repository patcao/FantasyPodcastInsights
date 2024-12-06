from typing import Optional
import pandas as pd
from src.constants import DateLike
from src.data_utils import load_clean_scores


class PlayerUtil:
    def __init__(self, seasons: Optional[list[str]] = None):
        df = load_clean_scores(seasons=seasons)
        self.df = df

    def players_for_date(self, date: DateLike) -> pd.DataFrame:
        """
        Get a list of players for an exact date.

        Parameters
        ----------
        date : DateLike
            The exact date for the query.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing unique players and their associated team for the given date,
            with columns: "personId", "personName", "teamName".
        """
        date_df = self.df[self.df["game_date"] == pd.to_datetime(date)]
        return date_df[["personId", "personName", "teamName"]].drop_duplicates()

    def players_for_date_range(
        self, begin_date: DateLike, end_date: DateLike
    ) -> pd.DataFrame:
        """
        Get a list of players for a date range.

        Parameters
        ----------
        begin_date : DateLike
            The starting date for the query.

        end_date : DateLike
            The exclusive ending date for the query.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing unique players and their associated team for the given date range,
            with columns: "personId", "personName", "teamName".
        """
        date_df = self.df[
            (self.df["game_date"] >= pd.to_datetime(begin_date))
            & (self.df["game_date"] < pd.to_datetime(end_date))
        ]
        return date_df[["personId", "personName", "teamName"]].drop_duplicates()

    def player_minute_stats(self, season: Optional[str] = None) -> pd.DataFrame:
        """
        Get aggregated player minute stats for each season.

        Returns
        -------
        pd.DataFrame
            A DataFrame with average minutes played per game, the number of games with over 5 minutes played,
            and total games per season for each player. Columns include:
            - "season_year"
            - "personId"
            - "personName"
            - "avg_minutes_per_game"
            - "games_over_5_minutes"
            - "total_games"
        """
        result = self.df
        if season is not None:
            result = result[result["season_year"] == season]

        result = (
            result.groupby(["season_year", "personId", "personName"])
            .agg(
                avg_minutes_per_game=("minutes", "mean"),
                games_over_5_minutes=("minutes", lambda x: (x > 5).sum()),
                total_games=("gameId", "nunique"),
            )
            .reset_index()
        )
        return result

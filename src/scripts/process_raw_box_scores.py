import os
import re

import numpy as np
import pandas as pd

from src.data_utils import load_and_filter

INPUT_DIRECTORY = "data/raw"
OUTPUT_DIRECTORY = "data/processed"

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


def normalize_name(name: str) -> str:
    """
    Normalizes a name  by converting them to lowercase and removing unnecessary characters or whitespace.
    """
    name = name.lower().strip()
    # Replace multiple spaces with a single space
    name = re.sub(r"\s+", " ", name)
    # Keeps letters, numbers, spaces, apostrophes, and hyphens
    name = re.sub(r"[^\w\s\'\-]", "", name)

    return name


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


def main():
    ctx = ("20100101", "20240501")

    r1 = load_and_filter(
        f"{INPUT_DIRECTORY}/regular_season_box_scores_2010_2024_part_1.csv", ctx
    )
    r2 = load_and_filter(
        f"{INPUT_DIRECTORY}/regular_season_box_scores_2010_2024_part_2.csv", ctx
    )
    r3 = load_and_filter(
        f"{INPUT_DIRECTORY}/regular_season_box_scores_2010_2024_part_3.csv", ctx
    )

    df = pd.concat([r1, r2, r3])
    df = preprocess_box_scores(df)

    df = df.assign(
        projectedFantasyPoints=df.groupby("personId")[
            "fantasyPoints"
        ]  # Group by personId
        .apply(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    df = df.assign(
        outperformed=df["fantasyPoints"] > df["projectedFantasyPoints"],
        outperformed_5=df["fantasyPoints"] - df["projectedFantasyPoints"] > 5,
        outperformed_10=df["fantasyPoints"] - df["projectedFantasyPoints"] > 10,
    )

    df = df.assign(
        outperform_next=df.groupby("personId")["outperformed"].shift(-1),
        outperform_next_5=df.groupby("personId")["outperformed_5"].shift(-1),
        outperform_next_10=df.groupby("personId")["outperformed_10"].shift(-1),
    )
    df["injured"] = np.where(~df.comment.isnull(), 1, 0)
    df["injured_next"] = df.groupby("personId")["injured"].shift(-1)
    df["weighted_injured"] = (
        0.95 * df["injured"]
        + 0.07 * df["injured_next"]
        + np.random.normal(0, 0.05, size=len(df))
    )

    df = df.dropna(
        subset=[
            "outperform_next",
            "outperform_next_5",
            "outperform_next_10",
            "injured_next",
        ]
    )

    df = df.assign(
        outperformed=df["outperformed"].astype(int),
        outperform_next=df["outperform_next"].astype(int),
    )

    df = df.reset_index(drop=True)

    df["personName"] = df["personName"].apply(normalize_name)
    df["teamName"] = df["teamName"].apply(normalize_name)

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    df.to_parquet(
        f"{OUTPUT_DIRECTORY}/regular_season_box_scores.pq",
        index=False,
        compression="snappy",
    )

    game_date_range = (str(df["game_date"].min()), str(df["game_date"].max()))
    print("--------Successfully Processed--------")
    print(
        f"Time Range: {game_date_range} \nGames: {len(df['gameId'].unique())} \nTeams: {len(df['teamId'].unique())} \nPlayers: {len(df['personId'].unique())}"
    )
    print()


if __name__ == "__main__":
    main()

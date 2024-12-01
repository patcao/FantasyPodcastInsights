import pandas as pd
from data_utils import load_and_filter, preprocess_box_scores
import os

INPUT_DIRECTORY = "data/raw"
OUTPUT_DIRECTORY = "data/processed"


def main():
    ctx = ("20190101", "20240501")

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
    df = df.assign(outperformed=df["fantasyPoints"] > df["projectedFantasyPoints"])
    df = df.reset_index(drop=True)

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

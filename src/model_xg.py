import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# lag_features = [
#     "minutes",
#     "fieldGoalsAttempted",
#     "fieldGoalsPercentage",
#     "threePointersAttempted",
#     "threePointersPercentage",
#     "freeThrowsAttempted",
#     "freeThrowsPercentage",
#     "reboundsDefensive",
#     "reboundsTotal",
#     "assists",
#     "steals",
#     "blocks",
#     "turnovers",
#     "foulsPersonal",
#     "points",
# ]

# select_features = ["plusMinusPoints"]

def add_lagged_features(df: pd.DataFrame, feature_names: list[str], max_lag: int):
    assign_map = {}

    for target_col in feature_names:
        for i in range(1, max_lag + 1):
            assign_map[f"{target_col}_lag_{i}"] = df.groupby("personId")[target_col].shift(
                i
            )

    return df.assign(**assign_map), list(assign_map.keys())


def train_xgboost_model(model, train_df,):
    pass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.data_utils import add_lagged_features


def train_and_plot(train, test, feature_names, target_name):
    X_train, X_test = train[feature_names], test[feature_names]
    y_train, y_test = train[target_name], test[target_name]

    # ---- Train XGBoost model ----
    # model = XGBRegressor(
    #     n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    # )
    model = XGBClassifier(objective="binary:logistic", random_state=42)
    model.fit(X_train, y_train)

    # ---- Evaluate the model ----
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # 5-day moving average performance
    print(
        "Sliding Window MAE:",
        np.mean(np.abs(y_test - test["projectedFantasyPoints"])),
    )
    print("Model MAE:", np.mean(np.abs(y_test - y_pred)))

    result_dct = {
        "model": model,
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return result_dct

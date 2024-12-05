import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime
from typing import Union, Optional, TypeAlias
import pandas as pd
import numpy as np
from pathlib import Path
from src.constants import DateLike


def get_repo_root():
    """Find the root of the repository based on the location of the .git directory."""
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.root:
        if (current_dir / ".git").is_dir():  # Check for .git folder
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError("Repository root not found (no .git directory)")


def compute_auc_roc(true_labels, predicted_labels) -> float:
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)

    # Compute AUC
    auc = roc_auc_score(true_labels, predicted_labels)
    print("AUC:", auc)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random chance line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return auc


def get_nba_season(date: DateLike) -> str:
    # Ensure the input date is in datetime format
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    elif isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    year = date.year
    # NBA seasons start in October and end in June the following year
    if (
        date.month >= 10
    ):  # From October to December, it's the current year's season start
        return f"{year}-{str(year + 1)[-2:]}"
    else:  # From January to September, it's the previous year's season
        return f"{year - 1}-{str(year)[-2:]}"

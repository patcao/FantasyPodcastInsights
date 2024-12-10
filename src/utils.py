from datetime import datetime
from pathlib import Path
from typing import Optional, TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.constants import DateLike


def get_repo_root():
    """Find the root of the repository based on the location of the .git directory."""
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.root:
        if (current_dir / ".git").is_dir():  # Check for .git folder
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError("Repository root not found (no .git directory)")


def combine_chunks_with_overlap(chunks: list[str], overlap_length: int):
    """
    Combine chunks while considering overlap

    Args:
    - chunks: List of text chunks.
    - overlap_length: Number of overlapping characters between consecutive chunks.

    Returns:
    - List of combined chunks.
    """
    combined_chunks = []
    current_chunk = chunks[0]  # Start with the first chunk

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]

        # Check if the overlap matches; combine the chunks
        if (
            current_chunk[-overlap_length:].strip()
            == next_chunk[:overlap_length].strip()
        ):
            current_chunk += next_chunk[overlap_length:]  # Add the non-overlapping part
        else:
            # Overlap doesn't match; finalize the current chunk and start a new one
            combined_chunks.append(current_chunk)
            current_chunk = next_chunk

    # Add the final chunk
    combined_chunks.append(current_chunk)

    return combined_chunks


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

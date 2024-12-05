# Fantasy Podcast Insights

## Overview

Fantasy basketball prediction models have traditionally relied on structured data like box scores, player stats, and game metrics. However, qualitative insights from expert analyses and player discussions, particularly from basketball podcasts, remain underutilized. 

This project leverages **large language models (LLMs)** to bridge that gap, combining structured statistics with unstructured podcast insights to identify sleeper picks—players likely to exceed their projected fantasy points in upcoming games.

### Key Features:
1. **Structured Data Analysis**: Utilizes box scores and historical player stats.
2. **Unstructured Insights**: Extracts contextually relevant features (e.g., sentiment, injury updates, team dynamics) from podcast transcriptions.
3. **Sleeper Identification**: Helps fantasy basketball managers identify undervalued players.
4. **LLM Integration**: Demonstrates the effectiveness of fine-tuned LLMs in sports prediction, an area traditionally dominated by statistical models.

## Repository Structure

```plaintext
FantasyPodcastInsights/
├── checkpoints/         # Model weights and checkpoints
├── configs/             # Configuration files (e.g., hyperparameters, paths)
├── data/                # Raw and processed data
├── notebooks/           # Jupyter notebooks for exploration and prototyping
├── src/                 # Core source code
│   ├── scripts/         # Standalone scripts for data processing and model training
├── environment.yaml     # Conda environment definition file
├── pyproject.toml       # Project configuration for reproducibility
├── LICENSE              # License information
├── README.md            # Project documentation
```

## Setup
1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/FantasyPodcastInsights.git
   cd FantasyPodcastInsights
   ```
2. **Setup Conda Environment**:
   ```
   conda env create -f environment.yaml
   conda activate FantasyPodcastInsights
   ```

## Cleaning Structured Data
```
python -m src.scripts.process_raw_box_scores
```
This will clean the box scores in data/raw and create a parquet file at data/processed/regular_season_box_scores.pq

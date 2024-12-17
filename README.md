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
├── data/                # Raw and processed data
├── notebooks/           # Jupyter notebooks for exploration and prototyping
├── src/                 # Core source code
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
   conda env create -f environment.yaml --name py310-fantasy
   conda activate py310-fantasy
   ```

3. **Ruff and Isort**:
This project using ruff formatter 
```
ruff format src/
```
and isort:
```
isort src/
```

## Cleaning Structured Data
```
python -m src.scripts.process_raw_box_scores
```
This will clean the box scores in data/raw and create a parquet file at data/processed/regular_season_box_scores.pq

## Running the Scraper
This notebook scrapes RSS feeds for new podcasts and downloads them.
```
notebooks/transcription-notebooks/NBA-Podcast-Scraper.ipynb
```

## Running the Transcription Model
This notebook takes mp3 files and transcribes them into text and storing the output
in the data/raw directory.
```
notebooks/GCP-Speech-To-Text-V2-Configuration.ipynb
```

## Evaluate NER Model
This noteobok, shows our evaluation of the NER portion of the pipeline.
```
notebooks/Evaluate-NER.ipynb
```

## Evaluate LLM Extracted Features
This notebook is used to evaluate LLM extracted features with the XGBoost Classifier
```
notebooks/Evaluate-XGBoost.ipynb
```

This notebook is used to evaluate LLM extracted features with the LSTM Classifier
```
notebooks/Evaluate-LSTM.ipynb
```
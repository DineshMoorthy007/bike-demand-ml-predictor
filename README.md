# Bike Rental Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-00599C)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that predicts hourly bike rental demand using the UCI Bike Sharing Dataset and serves predictions through a Streamlit dashboard.

## Project Overview

This project builds a complete ML workflow for bike rental forecasting:

- Data preprocessing pipeline for feature engineering and scaling
- Model training with XGBoost and hyperparameter tuning
- Model artifact persistence for reuse in inference
- Interactive Streamlit app for real-time demand prediction
- Dockerized deployment for consistent, portable execution

Core workflow:

- [data_pipeline.py](data_pipeline.py): Loads and preprocesses dataset features
- [train.py](train.py): Trains and tunes the regressor, then saves the model
- [app.py](app.py): Streamlit dashboard for user-driven predictions
- [models/xgb_bike_model.joblib](models/xgb_bike_model.joblib): Trained model artifact generated locally (not committed)

### Artifact Policy (User Directive)

- The [models/](models) directory is intentionally ignored in [.gitignore](.gitignore).
- Do not commit trained model files to GitHub.
- Every user must run training locally after cloning the project:

    python train.py

## Tech Stack

- Python 3.10+
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Docker

## Dataset Setup (Required)

This repository does not include the dataset file. You must download it locally before training or inference.

### Source

- UCI Bike Sharing Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

### Option 1: Manual setup

1. Download the dataset ZIP from UCI.
2. Extract the archive.
3. Copy [hour.csv](data/hour.csv) into the project [data](data) folder.
4. Final required path:

       data/hour.csv

### Option 2: PowerShell commands (Windows)

Run these from the project root:

    New-Item -ItemType Directory -Force data
    Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip" -OutFile "data/bike-sharing-dataset.zip"
    Expand-Archive -Path "data/bike-sharing-dataset.zip" -DestinationPath "data" -Force

After extraction, confirm [hour.csv](data/hour.csv) exists in [data](data).

## Usage

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Train the model

    python train.py

This creates:

- [models/xgb_bike_model.joblib](models/xgb_bike_model.joblib)

### 3. Run the Streamlit app locally

    streamlit run app.py

Open the dashboard at:

- http://localhost:8501

### 4. Build Docker image

Use this exact command:

    docker build -t bike-predictor .

### 5. Run Docker container

Use this exact command:

    docker run -p 8501:8501 bike-predictor

Then open:

- http://localhost:8501

## Notes

- Ensure the dataset file exists at [data/hour.csv](data/hour.csv).
- The model file [models/xgb_bike_model.joblib](models/xgb_bike_model.joblib) is expected to be created locally because [models/](models) is gitignored.
- Run training first after clone: python train.py.
- If you retrain, the app will load the latest saved model automatically.

## License

This project is licensed under the MIT License.

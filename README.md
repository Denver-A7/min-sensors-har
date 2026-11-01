# How Few Sensors Does a Wearable Need?

A Streamlit demo for the ME 131 project at SJSU. It shows how well simple models can recognize 6 daily activities with fewer sensor features. It uses the public UCI HAR dataset.

## Install

```
python -m pip install -r requirements.txt
```

Tested with Python 3.13 and scikit-learn 1.6.1. The saved models need scikit-learn 1.6.1.

## Run

```
python -m streamlit run Home.py
```

## Folders the app needs

The app only reads these folders. Please keep them next to `Home.py`.

- `results/` (result tables and `run_status.json`)
- `data_out/` (feature groups)
- `models/` (saved models)
- `demo_data/` (demo test windows)

The `.streamlit/config.toml` file sets the colors of the app.

## Warning: real or test data

The numbers in the app are only real when `results/run_status.json` says `true` for that file. If a file says `false`, the page shows a TEST DATA banner. Do not use those numbers as our results.

## Pages

- **Trade-off Explorer:** compare accuracy and number of features for each model and sensor group.
- **Try a Prediction:** pick a test window and see what a model predicts.
- **New Person Test:** see how the models do on a person they never saw.
- **Confusion Explorer:** see which activities get mixed up.
- **Sensor Tiers:** compare sensor choices by accuracy and power use.
- **About and Limits:** how the project was done, its limits, sources and the team.

## Data source

UCI Machine Learning Repository. Human Activity Recognition Using Smartphones (dataset 240). https://archive.ics.uci.edu/dataset/240

## Team

Andrew Rodricks, Ayham Abushanab, Charlie Morrow, Raiden De Luna, Denver Auncion.

Course: ME 131, Professor Yunjian (Jojo) Qiu, SJSU.

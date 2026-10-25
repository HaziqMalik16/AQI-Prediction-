# Global Air Pollution: AQI Prediction System

Regression project for **AIC354 Machine Learning Fundamentals**, Assignment 3 (Task 2).

| | |
|---|---|
| **Student** | Haziq Mahmood (FA24-BSE-049) |
| **Instructor** | Dr. Rao Muhammad Adeel Nawab |
| **Institution** | COMSATS University Islamabad, Lahore Campus, Fall 2026 |
| **Live demo** | (paste your Streamlit link here after deploying) |

## What the system does

| | |
|---|---|
| **Problem** | Estimate a city's overall AQI when its PM2.5 reading is not available |
| **Inputs** | CO, Ozone and NO2 AQI values, and the country (175 countries). PM2.5 is deliberately left out |
| **Output** | `AQI Value` (overall Air Quality Index, 0 to 500) |
| **Data** | Global Air Pollution dataset, 23,463 cities; 428 incomplete rows removed, 23,035 used |
| **Evaluation** | 10-fold cross validation, RMSE (plus MAE and R2), 5 algorithms |
| **Algorithms** | Ridge, Lasso, Support Vector Regression, Extra Trees, Histogram Gradient Boosting |
| **Best model** | Histogram Gradient Boosting: mean RMSE about 29.5 AQI points, R2 about 0.72 |
| **Simple rules** | Always predicting the average AQI: RMSE about 56.3. Largest visible value: about 62.5. All 5 algorithms beat both |
| **Extra experiment** | Same comparison with PM2.5 added: best is Extra Trees, RMSE about 2.16 |

Why PM2.5 is left out: the overall AQI is by definition the highest pollutant value (99.4% of cities), and PM2.5 is the highest in about 78% of them. With PM2.5 the task is almost "pick the largest number". See Step 3.3 of the notebook.

## Files

```
aqi_project/
├── aqi_prediction_kfold_full_dataset_10fold.ipynb   notebook (run on Google Colab)
├── global_air_pollution_dataset.csv                 dataset
├── aqi_prediction_model.pkl                         trained model (Histogram Gradient Boosting pipeline)
├── model_info.json                                  input ranges, countries, metrics (used by the app)
├── app.py                                           Streamlit web app
├── requirements.txt                                 packages for the deployed app
└── README.md
```

## Run the notebook on Google Colab

1. Open https://colab.research.google.com and choose **File > Upload notebook**.
2. **Runtime > Run all**.
3. When Step 2.1 asks, upload `global_air_pollution_dataset.csv`.

The notebook recreates `aqi_prediction_model.pkl` and `model_info.json`. Running everything takes roughly 15 to 25 minutes (SVR and Extra Trees are the slowest).

## Run the web app locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a **public GitHub repository** and upload the files above (`app.py` and `requirements.txt` at the top level).
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Select the repository, branch `main`, main file `app.py`. In *Advanced settings* choose **Python 3.12**.
4. Click **Deploy** and copy the live URL.
5. Paste the URL into the **first cell of the notebook** and into the table above.
6. Test the live app (predictions are rough, typical error about 30 AQI points):
   - Germany, CO 1, Ozone 30, NO2 2 gives about **44** (Good)
   - Germany, CO 1, Ozone 130, NO2 2 gives about **115** (Unhealthy for Sensitive Groups)
   - Pakistan, CO 3, Ozone 40, NO2 5 gives about **135** (Unhealthy for Sensitive Groups)

The model was trained with **scikit-learn 1.8.0**, which is pinned in `requirements.txt`. If the saved model cannot be loaded in the cloud, `app.py` re-creates the same model from the CSV.

## Disclaimer

Educational project. The predictions are rough statistical estimates, not official air-quality measurements.

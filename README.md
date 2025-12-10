# Hourly Electricity Load Forecasting for BEMS

This project is for the EF2039 AI Programming Term Project 02.  
The goal is to develop a machine learning model that predicts hourly
electricity load, which is an essential function of Building Energy
Management Systems (BEMS).

## 1. Dataset

- Source: Kaggle (continuous load dataset with weather and calendar information)
- Target variable: `nat_demand` (national electricity demand in MW)
- Main input features:
  - Time features: hour, day of week, month, is_weekend
  - Weather features: T2M_toc, QV2M_toc, TQL_toc, W2M_toc
  - Demand history features: lag1, lag24, rolling means (3h, 24h)
  - Calendar features: holiday, school

The raw CSV file is stored in `data/continuous_dataset.csv`.

## 2. Project Structure

```text
EF2039_Proj02_20240516_ChaewonLee/
 ├── data/
 │    └── continuous_dataset.csv
 ├── src/
 │    └── load_forecasting.py
 ├── results/
 │    └── hourly_load_prediction.png   # created after running the script
 ├── requirements.txt
 └── README.md

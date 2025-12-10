# Hourly Electricity Load Forecasting for BEMS

This project is the final assignment for **EF2039 AI Programming Term Project 02**.  
The goal is to build a machine learning pipeline that predicts **hourly national electricity load**, an essential function of **Building Energy Management Systems (BEMS)**.

The project includes:
- Data preprocessing  
- Feature engineering  
- Model training and performance comparison  
- Visualization of prediction results  

---

## 1. Dataset

- **Source:** Kaggle (continuous national electricity demand dataset with weather & calendar variables)
- **Target variable:** `nat_demand` (national electricity load in MW)
- **Key input features**
  - **Time features:** hour, day of week, month, is_weekend  
  - **Weather features:** T2M_toc, QV2M_toc, TQL_toc, W2M_toc  
  - **Demand history:** lag1, lag24, rolling means (3h, 24h)  
  - **Calendar features:** holiday, school  

The raw CSV file is stored in: `data/continuous_dataset.csv`.

---

## 2. Project Structure

EF2039_Proj02_ChaewonLee/
│── data/
│   └── continuous_dataset.csv
│
│── src/
│   └── load_forecasting.py
│
│── results/
│   └── hourly_load_prediction.png
│
├── requirements.txt
└── README.md


---

## 3. Installation & Execution

### 1) Install required packages
```bash
pip install -r requirements.txt

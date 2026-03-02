# ML-Based Vehicle Maintenance Prediction

[cite_start]This project implements a classical machine learning workflow to predict vehicle maintenance requirements using historical usage logs and sensor telemetry[cite: 1, 3]. [cite_start]By shifting the maintenance strategy from reactive to predictive, the system aims to minimize unexpected breakdowns and high operational costs[cite: 5, 9].

## 📊 Dataset Overview
[cite_start]The model is trained on a dataset containing **40,000 vehicle records**[cite: 4, 11].
* [cite_start]**Numerical Features:** Usage Hours, Engine Temperature (Celsius), Tire Pressure, Oil Quality, Battery Voltage, Vibration Level, and Maintenance Cost[cite: 12, 13].
* [cite_start]**Categorical Features:** Vehicle Type (Car, Truck, Bus) and Brake Condition (Good, Fair, Poor)[cite: 14].
* [cite_start]**Target Variable:** Maintenance Required (Binary: '1' for high risk, '0' for safe)[cite: 15].

## ⚙️ Methodology
* [cite_start]**Preprocessing:** Standardized anomalous temperature readings, handled missing values via median/mode imputation, and applied `StandardScaler` for normalization[cite: 122, 124, 125].
* [cite_start]**EDA Highlights:** Analysis revealed that **Engine Temperature** has the highest correlation (0.52) with failure risk, while **Oil Quality** shows a strong negative correlation (-0.35)[cite: 21, 22].
* [cite_start]**Algorithms:** Evaluated **Logistic Regression** as a robust baseline and **Decision Tree** classifiers to capture non-linear relationships[cite: 130, 131].
* [cite_start]**Optimization:** Conducted hyperparameter tuning for the Decision Tree, identifying an optimal `max_depth` of 11 to prevent overfitting[cite: 163, 164].

## 📈 Performance Results
[cite_start]Evaluated on an 8,000-sample test split[cite: 134]:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **86.84%** | **86.37%** | **83.36%** | **84.84%** |
| Decision Tree (Tuned) | 82.89% | 82.73% | 77.42% | 79.99% |

> [cite_start]**Key Finding:** Logistic Regression is the recommended architecture because it achieved the highest **Recall** (83.36%), which is critical for operational safety to avoid missing actual breakdowns[cite: 135, 140].
# Hospital Readmission Prediction

Can we predict which diabetic patients will be readmitted to the hospital 
within 30 days of discharge? That's the question this project tries to answer.

Using a real clinical dataset of 100,000+ patient records from 130 US hospitals,
I built a machine learning pipeline that combines unsupervised and supervised 
learning to flag high-risk patients before they leave the hospital.

## What I did

- Cleaned and preprocessed 50 features including demographics, diagnoses,
  medications, and hospital visit history
- Used K-Means clustering to segment patients into 4 groups based on 
  clinical similarity
- Trained a Random Forest classifier to predict 30-day readmission,
  using cluster labels as an engineered feature
- Handled class imbalance (only 11% of patients were readmitted) using 
  SMOTE and balanced class weights
- Tuned the decision threshold to improve recall from 4% to 37%

## Results

| Model | AUC-ROC | Recall (readmitted) |
|---|---|---|
| Logistic Regression (baseline) | 0.556 | 12% |
| Random Forest (default) | 0.582 | 4% |
| Random Forest (tuned) | 0.638 | 37% |

## Key findings

The strongest predictors of readmission were number of lab procedures,
number of medications, time spent in hospital, age, and prior inpatient visits.
These are all things a hospital knows before discharging a patient, which means
this model could realistically be used to trigger extra follow-up care for 
high-risk patients.

## Tools used

Python, scikit-learn, pandas, Plotly, imbalanced-learn

## Dataset

UCI Machine Learning Repository: Diabetes 130-US Hospitals (1999-2008)  
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

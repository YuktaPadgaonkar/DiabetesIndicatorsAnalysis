# Diabetes Prediction Using Machine Learning

> Predicting diabetes risk using demographic, lifestyle, and health indicators from the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset.

**Course:** CMPE 255 – Data Mining  
**Institution:** San José State University

---

## 📌 Project Overview

This project focuses on predicting diabetes risk using demographic, lifestyle, and health indicators derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset. By applying data preprocessing, feature engineering, class imbalance handling, and multiple machine learning models, the project aims to identify key risk factors and build reliable predictive models for diabetes.

---

## 🎯 Objectives

- Clean and preprocess large-scale public health survey data
- Perform exploratory data analysis (EDA) and feature engineering
- Address severe class imbalance in diabetes outcomes
- Train and compare multiple machine learning models
- Evaluate models using robust performance metrics
- Identify the most influential predictors of diabetes

---

## 📂 Repository Structure

```
├── CMPE-255_final_project.py        # End-to-end ML pipeline script
├── final_presentation.ipynb        # Final presentation notebook
├── Yukta_EDAandFeatureEngineering.ipynb
├── Yukta_FeatureSelection.ipynb
├── Yukta_Catboost.ipynb
├── aconvert_brfss_2024.py          # SAS (.XPT) to CSV conversion script
├── Report_DataMining_Group8.pdf    # Final project report
└── README.md
```

---

## 📊 Dataset

### Data Source

**Source:** CDC – Behavioral Risk Factor Surveillance System (BRFSS)

**Official Data Portal:** [https://www.cdc.gov/brfss/annual_data/annual_2024.html](https://www.cdc.gov/brfss/annual_data/annual_2024.html)

### Original Format

- **Format:** .XPT (SAS Transport file)

### Conversion to CSV

The BRFSS dataset is originally released in SAS (.XPT) format. To enable analysis using Python and modern machine learning libraries, we used a custom Python script (`aconvert_brfss_2024.py`) to convert the .XPT file into a compressed .CSV format.

**Key steps in the conversion process:**
- Loaded the .XPT file using `pyreadstat`
- Applied tolerant character encoding to avoid decoding errors
- Exported the dataset as a compressed `.csv.gz` file for efficiency

The converted CSV file was then used for all preprocessing, analysis, and modeling steps in this project.

### Dataset Characteristics

- **Initial Size:** ~457,000 rows × 301 columns
- **Final Dataset Used for Modeling:**
  - 51 curated features
  - Binary target variable: `HasDiabetes`

The dataset contains extensive information on:
- Demographics
- Lifestyle habits (smoking, drinking, exercise)
- Physical and mental health indicators
- Chronic health conditions

All included variables and preprocessing decisions are documented in detail in the final project report.

> ⚠️ **Note:** Due to dataset size and privacy considerations, the raw BRFSS dataset is not included in this repository.

---

## 🛠️ Methodology

### 1. Data Cleaning & Preprocessing

- Removed columns with more than 40% missing values
- Dropped irrelevant metadata and state-specific fields
- Handled missing values and ensured data consistency
- Encoded categorical features using:
  - Ordinal Encoding
  - One-Hot Encoding
- Scaled numerical features using Min-Max Scaling

### 2. Feature Engineering

- Created interaction features such as:
  - Age × BMI
  - Income × Education
- Reduced dimensionality while retaining interpretability

### 3. Class Imbalance Handling

The dataset is highly imbalanced (~13% diabetic cases).

**Techniques used:**
- SMOTE
- Borderline-SMOTE
- SMOTE-Tomek (final choice)
- Class-weighted models

---

## 🤖 Models Implemented

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost (feature importance & selection)
- Soft-Voting Ensemble Model

---

## 📈 Evaluation Metrics

- Accuracy
- F1-Score (weighted)
- ROC-AUC
- Confusion Matrix
- Classification Report

The ensemble model achieved the best overall balance between recall and precision.

---

## 🔍 Key Findings

**Top predictors of diabetes:**
- Age Group
- BMI
- Physical & Mental Health Status
- Smoking & Drinking Habits
- General Health Condition

Results align strongly with established medical research. Ensemble and boosting models consistently outperformed traditional classifiers.

---

## 🚀 How to Run the Project

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost lightgbm matplotlib pyreadstat
```

### 2. Run the Main ML Pipeline

```bash
python CMPE-255_final_project.py
```

### 3. Explore Notebooks

```bash
jupyter notebook
```

---

## 📌 Future Enhancements

- Incorporate longitudinal and pre-diagnosis data
- Expand predictions to pre-diabetes detection
- Build an interactive public health dashboard
- Periodic retraining with updated BRFSS datasets

---

## 👩‍💻 Contributors

**Group 8 – CMPE 255 (Data Mining)**  
San José State University

- Kavan Thaker
- Laxman Shah
- Yukta Padgaonkar

---

## 📜 License

This project is for academic and educational purposes only.  
Please cite appropriately if reused.

---

## 📚 References

- [CDC BRFSS Annual Survey Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
- Final project report: `Report_DataMining_Group8.pdf`

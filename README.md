# Diabetes Risk Prediction Using Deep Learning (MLP & LSTM)

A deep learning project comparing Multi-Layer Perceptron (MLP) and Long Short-Term Memory (LSTM) models for diabetes risk classification, built as an individual assignment for CT118-3-3-ODL (Optimisation and Deep Learning).

---

## 📌 Project Overview

This project builds and compares two neural network architectures — MLP and LSTM — for binary classification of diabetes risk using the BRFSS 2015 health survey dataset. The full pipeline covers EDA, data preprocessing, model building, hyperparameter tuning via random search, and threshold optimisation.

**Dataset:** Diabetes Health Indicators Dataset (BRFSS 2015, balanced subset) — 70,692 records, 21 features  
**Source:** [Kaggle — Alex Teboul](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)  
**Target Variable:** `Diabetes_binary` (0 = No diabetes, 1 = Diabetes or prediabetes)

---

## 📊 Results Summary

### Baseline Models (threshold = 0.50)

| Model | AUC | Accuracy | F1 |
|-------|-----|----------|----|
| MLP | 0.8237 | 0.7517 | 0.7720 |
| LSTM | 0.8255 | 0.7516 | 0.7697 |

### Tuned Models + Threshold Optimisation

| Model | Threshold | AUC | Accuracy | F1 |
|-------|-----------|-----|----------|----|
| MLP Tuned | 0.430 | 0.8248 | 0.7397 | 0.7774 |
| LSTM Tuned | 0.413 | 0.8252 | 0.7463 | 0.7792 |

The ROC curves of both tuned models nearly overlap (AUC ≈ 0.825), indicating comparable ranking ability. The LSTM achieves slightly fewer missed diagnoses (lower FN) and a marginally higher F1, making it the preferred model for screening use cases.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

## 📁 Project Structure

```
diabetes-deep-learning-mlp-lstm/
├── notebooks/
│   └── ODL.ipynb          # Full pipeline: EDA, preprocessing, model building, tuning, evaluation
└── README.md
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/limjaiyie/diabetes-deep-learning-mlp-lstm.git
cd diabetes-deep-learning-mlp-lstm
```

**2. Install dependencies**
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

**3. Download the dataset**

Download `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data) and place it in the project directory.

**4. Run the notebook**
```bash
jupyter notebook notebooks/ODL.ipynb
```

---

## 🔍 Pipeline Overview

1. **EDA** — Feature distribution analysis, duplicate detection, correlation heatmap
2. **Data Preprocessing** — Duplicate removal, outlier trimming, Min-Max normalisation, stratified 70/15/15 train/val/test split
3. **Model Building** — MLP baseline (Dense→BN→Dropout) and LSTM baseline
4. **Hyperparameter Tuning** — Random search: 12 trials for MLP, 10 trials for LSTM
5. **Threshold Optimisation** — Grid search on validation set to maximise F1
6. **Evaluation** — AUC, Accuracy, F1, confusion matrix on held-out test set

---

## 👤 Author

**Lim Jia Yie**  
BSc (Hons) Computer Science (Data Analytics)  
Asia Pacific University (APU) & De Montfort University (DMU)

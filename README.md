# Predicting Financial Inclusion in East Africa Using Machine Learning

A comparative study of classification and clustering algorithms applied to FinScope survey data from Kenya, Rwanda, Tanzania, and Uganda.

> **Course:** MSDA 9213 — Data Mining
> **Institution:** Adventist University of Central Africa (AUCA), Kigali, Rwanda
> **Date:** February 2026

---

## Overview

Financial inclusion remains a critical challenge in East Africa, where only **13.9%** of adults across Kenya, Rwanda, Tanzania, and Uganda have access to commercial bank accounts. This project applies machine learning to predict individual bank account ownership using demographic and socioeconomic data from FinScope surveys (2016–2018), covering **23,524 individuals** across four EAC countries.

---

## Dataset

- **Source:** [Zindi — Financial Inclusion in Africa Competition](https://zindi.africa/competitions/financial-inclusion-in-africa)
- **Training set:** 23,524 observations | **Test set:** 10,086 observations
- **Target variable:** `bank_account` (Yes / No)
- **Class imbalance:** 85.9% No / 14.1% Yes

### Features

| Feature | Description |
|---|---|
| `country` | Kenya, Rwanda, Tanzania, Uganda |
| `year` | Survey year (2016–2018) |
| `location_type` | Rural / Urban |
| `cellphone_access` | Yes / No |
| `household_size` | Number of people in household |
| `age_of_respondent` | Respondent's age |
| `gender_of_respondent` | Male / Female |
| `relationship_with_head` | Relationship to household head |
| `marital_status` | Marital status |
| `education_level` | Highest level of education attained |
| `job_type` | Employment / livelihood type |

---

## Methods

### Classification Algorithms

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline with L2 regularization |
| Random Forest | 200 trees, max depth 15 |
| Gradient Boosting | 200 estimators, depth 5, lr=0.1 |
| MLP Neural Network | 3 hidden layers (128→64→32), ReLU, Adam |

### Class Imbalance Strategies
- **Random Oversampling** — duplicates minority class to balance training data (18,819 → 32,338 samples)
- **Class Weighting** — `class_weight='balanced'` for LR and Random Forest

### Clustering Algorithms
- **K-Means** (K=4) — optimal K selected via Elbow method and Silhouette analysis
- **Hierarchical (Agglomerative)** — Ward's linkage on a 10,000-sample subset

---

## Results

### Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.874 | 0.649 | 0.227 | 0.336 | 0.836 |
| Random Forest | 0.886 | 0.662 | 0.384 | 0.486 | 0.866 |
| **Gradient Boosting** | **0.887** | **0.672** | **0.387** | **0.491** | **0.874** |
| MLP (Deep Learning) | 0.885 | 0.669 | 0.363 | 0.470 | 0.864 |

### Improved Model Performance (after class balancing & tuning)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.732 | 0.316 | 0.775 | 0.449 | 0.836 |
| **Random Forest** | **0.861** | **0.506** | **0.554** | **0.529** | **0.855** |
| Gradient Boosting | 0.814 | 0.410 | 0.736 | 0.527 | 0.867 |
| MLP (Deep Learning) | 0.817 | 0.402 | 0.622 | 0.489 | 0.816 |

### Clustering Performance

| Algorithm | Silhouette Score |
|---|---|
| K-Means (K=4) | 0.159 |
| Hierarchical (K=4) | 0.158 |

### Key Findings
- **Gradient Boosting** achieved the best baseline AUC-ROC (0.874); **Random Forest** achieved the best improved F1-Score (0.529)
- Recall improved by up to **+54.8 percentage points** (Logistic Regression) after oversampling
- Top predictors: **country**, **education level**, **job type**, **age**
- **Kenya** has the highest financial inclusion rate (26.3%); Rwanda the lowest (7.0%)
- Urban residents are ~3× more likely to hold bank accounts than rural residents

---

## Project Structure

```
├── Financial_Inclusion_EAC.ipynb   # Main Jupyter notebook (full pipeline)
├── Financial_Inclusion_Paper.pdf   # Research paper (compiled)
├── Financial_Inclusion_Paper.tex   # Research paper (LaTeX source)
├── submission.csv                  # Predictions on test set (Zindi format)
├── Datasets/
│   ├── Train.csv                   # Training data (23,524 rows)
│   └── Test.csv                    # Test data (10,086 rows)
└── figures/
    ├── fig01_target_distribution.png
    ├── fig02_Bank_account_by_country.png
    ├── fig03_Age_Distribution_by_country.png
    ├── fig04_Eduction_Level_VS_Account.png
    ├── fig05_Correlation.png
    ├── fig06_Job_VS_Bank_Account.png
    ├── fig07_Loaction_Cellphone.png
    ├── fig08_ROC_Curves.png
    ├── fig09_Confusion_Matrix.png
    ├── fig10_Feature_Importane.png
    ├── fig11_Comparison.png
    ├── fig12_Clustering.png
    ├── fig13_KMeans.png
    ├── fig14_Prediction_Test_Dataset.png
    ├── fig15_roc_curves_improved.png
    ├── fig16_Cluster_Profiles_Heatmap.png
    └── fig17_Financial_Inclusion_Rate_by_Cluseter.png
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/mbdidier/Predicting-Financial-Inclusion-in-EAC-Using-ML.git
   cd Predicting-Financial-Inclusion-in-EAC-Using-ML
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook Financial_Inclusion_EAC.ipynb
   ```

4. Run all cells in order — the notebook covers EDA, preprocessing, classification (baseline + improved), clustering, and test set prediction.

---

## Recommendations

1. **Education programs** — Individuals with no/primary education show the lowest inclusion rates; integrate financial literacy with basic education.
2. **Rural outreach** — Expand mobile banking and agent banking networks in rural areas.
3. **Mobile-first solutions** — Strong association between cellphone access and bank accounts; leverage this channel.
4. **Country-specific strategies** — Rwanda and Tanzania require tailored interventions compared to Kenya.
5. **Class-balanced modeling** — Always apply balancing techniques when predicting for underserved populations.

---

## References

- Demirguc-Kunt et al. (2018). *The Global Findex Database 2017.* World Bank Publications.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
- James et al. (2023). *An Introduction to Statistical Learning: with Applications in Python.* Springer.
- Zindi (2019). [Financial Inclusion in Africa Competition](https://zindi.africa/competitions/financial-inclusion-in-africa).

# 🔍 Mobile Transaction Fraud Detection

An ensemble machine learning system for detecting fraudulent mobile money transactions, built on the PaySim synthetic dataset. Combines **Random Forest**, **XGBoost**, **Logistic Regression**, and a **Neural Network** into a stacking ensemble achieving a PR-AUC of **0.9928**.

---

## 📊 Results

| Metric | Score |
|---|---|
| Precision | 0.97 |
| Recall | 0.99 |
| F1 Score | 0.98 |
| PR-AUC | 0.9928 |
| Fraud caught (test set) | 1,530 / 1,543 |
| False alarms | 48 / 553,038 |

---

## 📁 Project Structure

```
📁 fraud-detection/
  ├── transactions_train.csv     # Raw dataset (Kaggle)
  ├── notebook.ipynb             # Full pipeline — EDA to evaluation
  ├── app.py                     # Streamlit fraud detection app
  ├── fraud_ensemble.pkl         # Saved ensemble model
  ├── fraud_scaler.pkl           # Saved StandardScaler
  └── requirements.txt           # Python dependencies
```

---

## 🗂️ Dataset

- **Source:** [Fraud Detection — Bannour Chaker (Kaggle)](https://www.kaggle.com/datasets/bannourchaker/frauddetection)
- **Usability Score:** 8.82
- **Type:** PaySim synthetic mobile money transactions
- **Size:** 6,351,193 rows × 10 columns
- **Fraud rate:** 0.12% (7,717 fraud cases)

### Columns

| Column | Type | Description |
|---|---|---|
| `step` | int | Hour of transaction (1–744) |
| `type` | string | Transaction type |
| `amount` | float | Transaction amount |
| `nameOrig` | string | Sender account ID |
| `oldbalanceOrig` | float | Sender balance before |
| `newbalanceOrig` | float | Sender balance after |
| `nameDest` | string | Receiver account ID |
| `oldbalanceDest` | float | Receiver balance before |
| `newbalanceDest` | float | Receiver balance after |
| `isFraud` | int | Target label (0 = legit, 1 = fraud) |

---

## ⚙️ Pipeline Overview

### Step 1 — Exploratory Data Analysis
- Zero nulls and zero duplicates confirmed
- Fraud rate: **0.12%** — severe class imbalance identified
- Fraud exclusively in `TRANSFER` (0.72%) and `CASH_OUT` (0.17%)
- Fraud amounts are **8x larger** on average than legitimate transactions
- Origin accounts drained to zero in nearly all fraud cases

### Step 2 — Feature Engineering
Filtered dataset to `TRANSFER` and `CASH_OUT` only (6.35M → 2.76M rows).

Five new features created from EDA insights:

| Feature | Formula | Signal |
|---|---|---|
| `balance_diff_orig` | `oldbalanceOrig - newbalanceOrig` | How much left the sender |
| `balance_diff_dest` | `newbalanceDest - oldbalanceDest` | How much arrived at receiver |
| `balance_mismatch` | `balance_diff_orig - amount` | Debit vs transfer discrepancy |
| `orig_zeroed_out` | `1 if newbalanceOrig == 0` | Account fully drained |
| `amount_ratio_orig` | `amount / (oldbalanceOrig + 1)` | Fraction of balance moved |

Dropped `nameOrig`, `nameDest` (noise) and `isFlaggedFraud` (data leakage).
Final feature set: **12 features**. Split: **80% train / 20% test** (stratified).

### Step 3 — Handle Class Imbalance
Applied **SMOTE** to training data only — never the test set.

| | Before SMOTE | After SMOTE |
|---|---|---|
| Fraud cases | 6,174 | 2,205,974 |
| Legitimate cases | 2,205,974 | 2,205,974 |
| Fraud rate | 0.279% | 50.00% |

### Step 4 — Train Base Models

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 0.9751 | 0.9909 | 0.9830 |
| XGBoost | 0.9409 | 0.9909 | 0.9653 |
| Logistic Regression | 0.0832 | 0.9611 | 0.1532 |
| Neural Network (MLP) | 0.4610 | 0.9877 | 0.6286 |

> Logistic Regression and Neural Network used `StandardScaler` for feature scaling.

### Step 5 — Stacking Ensemble
Combined all four base models using `StackingClassifier` with a Logistic Regression meta-learner and `cv=5`.

```python
ensemble = StackingClassifier(
    estimators=[
        ('random_forest',        rf_model),
        ('xgboost',              xgb_model),
        ('logistic_regression',  lr_model),
        ('neural_network',       mlp_model),
    ],
    final_estimator = LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=False
)
```

### Step 6 — Evaluation
Evaluated on 553,038 unseen test transactions.

**Confusion Matrix:**

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 551,447 ✅ | 48 ❌ |
| **Actual Fraud** | 13 ❌ | 1,530 ✅ |

---

## 🚀 Running the App

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
joblib
numpy
pandas
scikit-learn
xgboost
imbalanced-learn
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Architecture

```
Input (12 features)
        │
        ├──► Random Forest     (n_estimators=200, class_weight='balanced')
        ├──► XGBoost           (n_estimators=200, learning_rate=0.1, max_depth=6)
        ├──► Logistic Regression (class_weight='balanced', max_iter=1000)
        └──► Neural Network    (hidden_layers=(64,32), activation='relu')
                │
                ▼
        Meta-Learner (Logistic Regression, cv=5)
                │
                ▼
        Fraud Probability (0.0 → 1.0)
```

---

## 🔬 How Fraud Works in This Dataset

Fraud exclusively occurs in two transaction types:

- **TRANSFER** — criminal drains victim account and routes funds to a mule account
- **CASH_OUT** — funds converted to physical cash via a mobile money agent

The five conditions that identify fraud:
1. Transaction type is `TRANSFER` or `CASH_OUT`
2. Origin account drained to zero (`newBalanceOrig = 0`)
3. Transaction amount is very large (average 8x legitimate)
4. Balance mismatch between sender and receiver
5. Amount close to the system limit of 10,000,000

> Our model catches fraud at the first digital hop — before physical cash is ever withdrawn.

---

## 📄 Documentation

Full step-by-step PDF reports are included in the `/docs` folder:

| Report | Contents |
|---|---|
| `step1_eda_report.pdf` | EDA findings and data insights |
| `step2_feature_engineering_report.pdf` | Feature creation and train/test split |
| `steps3_4_report.pdf` | SMOTE results and base model performance |
| `steps5_6_report.pdf` | Ensemble configuration and final evaluation |
| `fraud_mechanics_report.pdf` | How fraud works in mobile transactions |

---

## 👤 Author

Built as an academic machine learning project.  
Dataset by **Bannour Chaker** — [Kaggle](https://www.kaggle.com/datasets/bannourchaker/frauddetection)

---

## 📜 License

This project is for educational purposes only.

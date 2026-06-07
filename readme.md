# Heart Disease Risk Assessment

A machine learning pipeline that predicts the presence of heart disease from clinical indicators, paired with a Streamlit web app for interactive clinical decision support.

**Model:** Random Forest · **Dataset:** Cleveland Heart Disease (UCI) · **Interface:** Streamlit

---

## Project Structure

```
ASM-4/
├── notebooks/
│   ├── assignment4.ipynb          # Original exploration notebook
│   └── assignment4_clean.ipynb    # Final cleaned notebook (run this)
├── app/
│   ├── app.py                     # Streamlit application
│   ├── heart_model.pkl            # Trained Random Forest model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── feature_names.pkl          # Encoded feature column names
│   └── sample_patient.pkl         # Pre-loaded test patient
├── report/
│   └── report.pdf                 # Full written report
├── artifacts/
│   └── diagrams/                  # Figures and plots
└── requirements.txt
```

---

## Dataset Download

The project uses the **Cleveland Heart Disease** dataset from the UCI Machine Learning Repository.

1. Go to: https://archive.ics.uci.edu/dataset/45/heart+disease
2. Download `processed.cleveland.data`
3. Place it in the project root (same level as `notebooks/`):

```
ASM-4/
└── processed.cleveland.data
```

> **Note:** The file has no header row. The notebook assigns column names automatically — no manual editing required.

---

## Running the Notebook

### Prerequisites

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Steps

1. Make sure `processed.cleveland.data` is in the project root (see above).
2. Open the notebook:

```bash
jupyter notebook notebooks/assignment4_clean.ipynb
```

3. Run all cells top to bottom (**Kernel → Restart & Run All**).

The notebook will:
- Load and preprocess the Cleveland dataset
- Handle missing values and encode categorical features
- Train and evaluate a Random Forest classifier
- Save `heart_model.pkl`, `scaler.pkl`, `feature_names.pkl`, and `sample_patient.pkl` into the `app/` folder

> **Important:** Run the notebook at least once before launching the app, so the `.pkl` artifact files are generated.

---

## Running the Streamlit App

Make sure you are in the project root directory and the `.pkl` files exist in `app/`.

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501` in your browser.

### Using the App

- The sidebar form is **pre-populated** with a real test patient from the dataset.
- Adjust any clinical values (age, cholesterol, chest pain type, etc.) using the sidebar inputs.
- Click **Run Prediction** to generate a risk assessment.
- The main panel displays:
  - Prediction outcome (Disease Present / No Disease Detected)
  - Disease probability with a visual progress bar
  - Top 3 most important features driving the prediction
  - A plain-English clinical interpretation

---

## Feature Reference

| Feature    | Type        | Range / Values      | Description                          |
|------------|-------------|---------------------|--------------------------------------|
| `age`      | Numeric     | 20 – 80             | Age in years                         |
| `sex`      | Binary      | 0 / 1               | 0 = Female, 1 = Male                 |
| `cp`       | Categorical | 1 – 4               | Chest pain type                      |
| `trestbps` | Numeric     | 80 – 200 mmHg       | Resting blood pressure               |
| `chol`     | Numeric     | 100 – 600 mg/dl     | Serum cholesterol                    |
| `fbs`      | Binary      | 0 / 1               | Fasting blood sugar > 120 mg/dl      |
| `restecg`  | Categorical | 0 – 2               | Resting ECG results                  |
| `thalach`  | Numeric     | 70 – 210 bpm        | Maximum heart rate achieved          |
| `exang`    | Binary      | 0 / 1               | Exercise-induced angina              |
| `oldpeak`  | Numeric     | 0.0 – 6.2           | ST depression induced by exercise    |
| `slope`    | Categorical | 1 – 3               | Slope of peak exercise ST segment    |
| `ca`       | Numeric     | 0 – 3               | Major vessels coloured by fluoroscopy|
| `thal`     | Categorical | 3 / 6 / 7           | Thalassemia type                     |

---

## Dependencies

Key libraries (see `requirements.txt` for pinned versions):

- `scikit-learn` — model training and preprocessing
- `xgboost` — gradient boosting (explored in notebook)
- `imbalanced-learn` — SMOTE oversampling
- `shap` — feature importance explanations
- `streamlit` — web application
- `pandas`, `numpy`, `matplotlib`, `seaborn` — data handling and visualisation
- `tensorflow` — deep learning experiments (notebook)

---

## Authors

**Ahmed Bilal** — 23I-2581  

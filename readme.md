# Blood Pressure Determinants — Advanced Statistics Project

> **Course:** Advanced Statistics | FAST NUCES Islamabad  
> **Student:** Ahmed Bilal | Roll No. 23I-2581 | Section DS-A  
> **Dataset:** NHANES 1988–2018 (Kaggle / CDC)

---

## Abstract

This project investigates the socioeconomic, dietary, and lifestyle determinants of **systolic blood pressure (SBP)** using data from the National Health and Nutrition Examination Survey (NHANES) 1988–2018. Two predictive models are developed and compared — **Multiple Linear Regression (MLR)** and a **Random Forest Regressor** — to quantify the relative contribution of age, BMI, physical activity, dietary sodium intake, income level, and sedentary hours to SBP variation in a large, nationally representative U.S. adult cohort.

---

## Variables

| Variable | NHANES Code | Role | Description |
|----------|------------|------|-------------|
| Systolic BP | `VNAVEBPXSY` | **Dependent (Y)** | Average systolic blood pressure (mmHg) |
| Age | `RIDAGEYR` | Independent | Age in years (adults ≥ 18) |
| BMI | `BMXBMI` | Independent | Body Mass Index (kg/m²) |
| Sedentary Hours | `PAD680` | Independent | Daily sedentary time (hours) |
| Sodium Intake | `DRXTSODI` | Independent | Dietary sodium (mg/day) |
| Income Level | `INDFMPIR` | Independent | Family income-to-poverty ratio |
| Physical Activity | `PAQ650` | Independent | Vigorous physical activity (binary: 1=Yes, 0=No) |

---

## Key Results

| Metric | Multiple Linear Regression | Random Forest | Better |
|--------|---------------------------|---------------|--------|
| MSE | 242.05 | 238.13 | RF |
| RMSE | 15.56 mmHg | 15.43 mmHg | RF |
| R² (test) | 0.1885 | 0.2016 | RF |
| 5-Fold CV R² | 0.2165 ± 0.017 | 0.2147 ± 0.036 | Tie |
| Interpretability | High | Moderate | MLR |

> Run `src/analysis.py` to reproduce these results.

---

## Project Structure

```
Adv-Stat-Project/
│
├── README.md                         ← This file
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/                          ← Raw CSVs (gitignored; see data/README.md)
│   └── README.md                     ← Download instructions & variable dictionary
│
├── notebooks/
│   ├── 23i-2581.ipynb                ← Main exploratory analysis notebook
│   └── Ahmed_Bilal_Muhammad.ipynb    ← Alternative / partner notebook
│
├── src/
│   ├── analysis.py                   ← Full analysis pipeline (Tasks 01–05)
│   ├── extract_stats.py              ← Helper: extract summary statistics
│   └── fix_encoding.py               ← Helper: fix CSV encoding issues
│
├── outputs/
│   ├── Task01_SummaryStats.csv       ← Summary statistics table
│   ├── Task02_BoxPlots.png           ← Box & whisker plots
│   ├── Task03_ScatterGrid.png        ← Scatter plot grid (IVs vs SBP)
│   ├── Task05_PredVsActual.png       ← Predicted vs Actual (MLR & RF)
│   ├── Task05_Residuals.png          ← Residual analysis
│   ├── Task05_FeatureImportance.png  ← RF feature importances
│   └── Task05_ModelComparison.png    ← MSE / RMSE / R² bar charts
│
├── reports/
│   ├── report.tex                    ← LaTeX source
│   ├── Advanced_Statistics_Project.pdf ← Final compiled report
│   ├── Blood_Pressure_Literature_Review.docx
│   └── project-instructions.md
│
└── references/
    └── README.md                     ← APA citations for all 11 papers
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/ahmedbilalazhar/Adv-Stat-Project.git
cd Adv-Stat-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data

See [`data/README.md`](data/README.md) for instructions to download the NHANES CSVs from Kaggle and place them in `data/raw/`.

### 4. Run the analysis

```bash
python src/analysis.py
```

All output figures and tables will be saved to the `outputs/` folder.

---

## Output Visualizations

### Task 02 — Box & Whisker Plots
![Box Plots](outputs/Task02_BoxPlots.png)

### Task 03 — Scatter Plot Grid
![Scatter Grid](outputs/Task03_ScatterGrid.png)

### Task 05 — Model Comparison
![Model Comparison](outputs/Task05_ModelComparison.png)

### Task 05 — Predicted vs Actual
![Predicted vs Actual](outputs/Task05_PredVsActual.png)

---

## References

See [`references/README.md`](references/README.md) for the full list of 11 cited papers in APA 7th edition format.

---

## License

This project is submitted for academic purposes only.  
Dataset © U.S. Centers for Disease Control and Prevention (CDC) — Public Domain.

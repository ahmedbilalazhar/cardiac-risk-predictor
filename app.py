import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model          = joblib.load(os.path.join(BASE_DIR, 'heart_model.pkl'))
scaler         = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
feat_names     = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))
sample_patient = joblib.load(os.path.join(BASE_DIR, 'sample_patient.pkl'))

def get_raw_to_encoded(raw_vals):
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    cat_cols  = ['cp', 'restecg', 'slope', 'thal']
    raw_df = pd.DataFrame([raw_vals])
    enc_df = pd.get_dummies(raw_df, columns=cat_cols)
    enc_df = enc_df.reindex(columns=feat_names, fill_value=0)
    cols_to_scale = [c for c in cont_cols if c in enc_df.columns]
    enc_df[cols_to_scale] = scaler.transform(enc_df[cols_to_scale])
    return enc_df

st.set_page_config(
    page_title="Heart Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Serif+Display&display=swap');

:root {
    --navy:          #0D1F35;
    --blue:          #2563EB;
    --blue-dark:     #1D4ED8;
    --blue-faint:    #EFF6FF;
    --blue-border:   #BFDBFE;
    --red:           #DC2626;
    --red-faint:     #FEF2F2;
    --red-border:    #FECACA;
    --green:         #16A34A;
    --green-faint:   #F0FDF4;
    --green-border:  #BBF7D0;
    --slate-50:      #F8FAFC;
    --slate-100:     #F1F5F9;
    --slate-200:     #E2E8F0;
    --slate-300:     #CBD5E1;
    --slate-400:     #94A3B8;
    --slate-500:     #64748B;
    --slate-600:     #475569;
    --slate-700:     #334155;
    --slate-900:     #0F172A;
    --white:         #FFFFFF;
    --r-sm:          6px;
    --r-md:          10px;
    --r-lg:          14px;
    --shadow-xs:     0 1px 2px rgba(0,0,0,0.05);
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:     0 4px 16px rgba(13,31,53,0.09), 0 1px 4px rgba(0,0,0,0.05);
    --shadow-lg:     0 8px 28px rgba(13,31,53,0.13), 0 2px 6px rgba(0,0,0,0.06);
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
}
.stApp { background: var(--slate-50) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ════════════════════
   SIDEBAR
════════════════════ */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.25) !important;
}

/* Force all sidebar text light */
[data-testid="stSidebar"] *:not(button):not(svg):not(path) {
    color: #E2E8F0 !important;
}

/* ── THE FIX: target actual input/select value elements by testid ── */
[data-testid="stNumberInputField"] {
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    caret-color: #3B82F6 !important;
}
[data-testid="stNumberInputField"]::placeholder {
    color: rgba(203,213,225,0.40) !important;
    -webkit-text-fill-color: rgba(203,213,225,0.40) !important;
}

/* Selectbox displayed value */
[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[class*="singleValue"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[class*="placeholder"],
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] div {
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] label {
    font-size: 0.685rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: #6B8CAE !important;
    -webkit-text-fill-color: #6B8CAE !important;
    margin-bottom: 3px !important;
}

/* Number input wrapper */
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: var(--r-sm) !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div:focus-within,
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.20) !important;
}

/* Step +/- buttons */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.05) !important;
    border: none !important;
    color: #6B8CAE !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background: rgba(255,255,255,0.12) !important;
    color: #F1F5F9 !important;
}

/* Selectbox wrapper */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: var(--r-sm) !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    fill: #6B8CAE !important;
}

/* Run Prediction button */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: var(--blue) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.87rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    padding: 0.62rem 1rem !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 2px 12px rgba(37,99,235,0.40) !important;
    transition: background 0.14s, box-shadow 0.14s, transform 0.1s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--blue-dark) !important;
    box-shadow: 0 4px 18px rgba(37,99,235,0.55) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0) !important;
}

/* ════════════════════
   MAIN AREA
════════════════════ */
h1,h2,h3,h4,h5 {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--slate-900) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--white) !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-xs) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: var(--slate-600) !important;
}

/* Tables */
table { border-collapse: collapse !important; width: 100% !important; font-size: 0.84rem !important; }
thead tr th {
    background: var(--navy) !important;
    color: #94A3B8 !important;
    padding: 9px 13px !important;
    font-size: 0.685rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    text-align: left !important;
}
tbody tr td {
    padding: 8px 13px !important;
    border-bottom: 1px solid var(--slate-200) !important;
    color: var(--slate-700) !important;
}
tbody tr:last-child td { border-bottom: none !important; }
tbody tr:nth-child(even) td { background: var(--slate-50) !important; }
tbody tr:hover td { background: var(--blue-faint) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--slate-300); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--slate-400); }

/* ════════════════════
   CUSTOM COMPONENTS
════════════════════ */

/* Page header */
.ph {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding: 1.4rem 0 1.2rem;
    border-bottom: 1px solid var(--slate-200);
    margin-bottom: 1.5rem;
}
.ph-left {}
.ph-eye {
    font-size: 0.685rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 0.18rem;
}
.ph-title {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 1.75rem;
    font-weight: 400;
    color: var(--navy);
    line-height: 1.12;
    margin: 0;
}
.ph-sub {
    font-size: 0.82rem;
    color: var(--slate-400);
    margin-top: 0.28rem;
    font-weight: 400;
}
.ph-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--green-faint);
    color: var(--green);
    border: 1px solid var(--green-border);
    border-radius: 99px;
    padding: 0.28rem 0.75rem;
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    white-space: nowrap;
}

/* Sidebar logo */
.sb-logo {
    padding: 1.3rem 1.1rem 1.0rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.2rem;
}
.sb-logo-mark {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.25rem;
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
    line-height: 1;
    letter-spacing: -0.01em;
}
.sb-logo-sub {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #3D5A7A !important;
    -webkit-text-fill-color: #3D5A7A !important;
    margin-top: 5px;
}

/* Sidebar section headers */
.sb-sec {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.11em !important;
    text-transform: uppercase !important;
    color: #3D5A7A !important;
    -webkit-text-fill-color: #3D5A7A !important;
    margin: 1.1rem 0 0.45rem !important;
    padding-top: 0.85rem !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    display: block !important;
}
.sb-sec.first { margin-top: 0.15rem !important; padding-top: 0 !important; border-top: none !important; }

/* Result card */
.rc {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: var(--r-lg);
    padding: 1.4rem 1.5rem 1.25rem;
    box-shadow: var(--shadow-md);
}
.rc-lbl {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--slate-400);
    margin-bottom: 0.75rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--slate-200);
}
.rc-title-high {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--red);
    line-height: 1.1;
    margin-bottom: 0.55rem;
}
.rc-title-low {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--green);
    line-height: 1.1;
    margin-bottom: 0.55rem;
}
.badge-h {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--red-faint); color: var(--red);
    border: 1px solid var(--red-border);
    border-radius: var(--r-sm);
    padding: 0.32rem 0.8rem;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em;
}
.badge-l {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--green-faint); color: var(--green);
    border: 1px solid var(--green-border);
    border-radius: var(--r-sm);
    padding: 0.32rem 0.8rem;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em;
}

/* Probability bar */
.pb-wrap { margin-top: 1.1rem; }
.pb-header {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 6px;
}
.pb-label {
    font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--slate-400);
}
.pb-pct-h { font-size: 0.96rem; font-weight: 700; color: var(--red); letter-spacing: -0.02em; }
.pb-pct-l { font-size: 0.96rem; font-weight: 700; color: var(--green); letter-spacing: -0.02em; }
.pb-track { height: 6px; background: var(--slate-100); border-radius: 99px; overflow: hidden; }
.pb-fill-h { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #FCA5A5, #DC2626); }
.pb-fill-l { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #86EFAC, #16A34A); }

/* Stats strip */
.ss { display: flex; gap: 0.55rem; margin-top: 1.1rem; }
.sp {
    flex: 1; background: var(--slate-100);
    border: 1px solid var(--slate-200);
    border-radius: var(--r-sm);
    padding: 0.6rem 0.4rem; text-align: center;
}
.sp-val { font-size: 1.12rem; font-weight: 700; color: var(--navy); line-height: 1.1; letter-spacing: -0.02em; }
.sp-lbl { font-size: 0.60rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--slate-400); margin-top: 3px; }

/* Chart wrapper card */
.cc {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: var(--r-lg);
    padding: 1.4rem 1.5rem 1.0rem;
    box-shadow: var(--shadow-md);
}
.cc-lbl {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--slate-400);
    margin-bottom: 0.75rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--slate-200);
}

/* Explanation box */
.eb {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-left: 3px solid var(--blue);
    border-radius: 0 var(--r-md) var(--r-md) 0;
    padding: 1.0rem 1.25rem;
    margin-top: 1.1rem;
    box-shadow: var(--shadow-xs);
}
.eb.h { border-left-color: var(--red); }
.eb.l { border-left-color: var(--green); }
.eb-lbl { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: var(--slate-400); margin-bottom: 0.45rem; }
.eb-txt { font-size: 0.89rem; line-height: 1.70; color: var(--slate-600); }

/* Divider */
.hr-line { height: 1px; background: var(--slate-200); margin: 1.3rem 0; }

/* Empty state */
.es {
    background: var(--white); border: 1px solid var(--slate-200);
    border-radius: var(--r-lg); padding: 2.8rem 2rem;
    text-align: center; box-shadow: var(--shadow-xs);
}
.es-icon { font-size: 2.4rem; margin-bottom: 0.75rem; }
.es-title { font-family: 'DM Serif Display', serif; font-size: 1.18rem; font-weight: 400; color: var(--navy); margin-bottom: 0.35rem; }
.es-sub { font-size: 0.86rem; color: var(--slate-500); line-height: 1.65; }

/* Ref table heading */
.ref-heading {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--slate-500);
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
sp = sample_patient

def raw(key, fallback):
    v = sp.get(key, fallback)
    return v if not (isinstance(v, float) and np.isnan(v)) else fallback

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-mark">🫀 HeartGuard</div>
        <div class="sb-logo-sub">Clinical Risk Assessment · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-sec first">Patient Demographics</span>', unsafe_allow_html=True)
    age_val = st.number_input("Age (years)", min_value=20, max_value=80,
                              value=int(raw('age', 57)), step=1)
    sex_val = st.selectbox("Sex", options=[1, 0],
                           format_func=lambda x: "Male" if x == 1 else "Female",
                           index=0 if int(raw('sex', 1)) == 1 else 1)

    st.markdown('<span class="sb-sec">Cardiac Markers</span>', unsafe_allow_html=True)
    trestbps_val = st.number_input("Resting Blood Pressure (mmHg)",
                                   min_value=80, max_value=200,
                                   value=int(raw('trestbps', 130)), step=1)
    chol_val     = st.number_input("Serum Cholesterol (mg/dl)",
                                   min_value=100, max_value=600,
                                   value=int(raw('chol', 236)), step=1)
    thalach_val  = st.number_input("Max Heart Rate Achieved",
                                   min_value=70, max_value=210,
                                   value=int(raw('thalach', 150)), step=1)
    oldpeak_val  = st.number_input("ST Depression (oldpeak)",
                                   min_value=0.0, max_value=6.2,
                                   value=round(float(raw('oldpeak', 1.0)), 1),
                                   step=0.1, format="%.1f")
    fbs_val      = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                options=[0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No",
                                index=int(raw('fbs', 0)))
    exang_val    = st.selectbox("Exercise-Induced Angina",
                                options=[0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No",
                                index=int(raw('exang', 0)))

    st.markdown('<span class="sb-sec">Clinical Assessment</span>', unsafe_allow_html=True)
    cp_opts   = [1, 2, 3, 4]
    cp_labels = {1:"Typical Angina", 2:"Atypical Angina", 3:"Non-Anginal Pain", 4:"Asymptomatic"}
    cp_def    = int(raw('cp', 1))
    cp_val    = st.selectbox("Chest Pain Type", options=cp_opts,
                             format_func=lambda x: cp_labels[x],
                             index=cp_opts.index(cp_def) if cp_def in cp_opts else 0)

    re_opts   = [0, 1, 2]
    re_labels = {0:"Normal", 1:"ST-T Abnormality", 2:"LV Hypertrophy"}
    re_def    = int(raw('restecg', 0))
    restecg_val = st.selectbox("Resting ECG Result", options=re_opts,
                               format_func=lambda x: re_labels[x],
                               index=re_opts.index(re_def) if re_def in re_opts else 0)

    sl_opts   = [1, 2, 3]
    sl_labels = {1:"Upsloping", 2:"Flat", 3:"Downsloping"}
    sl_def    = int(raw('slope', 1))
    slope_val = st.selectbox("ST Segment Slope", options=sl_opts,
                             format_func=lambda x: sl_labels[x],
                             index=sl_opts.index(sl_def) if sl_def in sl_opts else 0)

    th_opts   = [3, 6, 7]
    th_labels = {3:"Normal", 6:"Fixed Defect", 7:"Reversible Defect"}
    th_def    = int(raw('thal', 3))
    thal_val  = st.selectbox("Thalassemia Type", options=th_opts,
                             format_func=lambda x: th_labels[x],
                             index=th_opts.index(th_def) if th_def in th_opts else 0)

    ca_val = st.number_input("Major Vessels (fluoroscopy, 0–3)",
                             min_value=0, max_value=3,
                             value=int(raw('ca', 0)), step=1)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", type="primary")

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="ph">
  <div class="ph-left">
    <div class="ph-eye">Clinical Decision Support</div>
    <div class="ph-title">Heart Disease Risk Assessment</div>
    <div class="ph-sub">Random Forest · Cleveland Heart Disease Dataset · Pre-populated with a real test patient</div>
  </div>
  <div class="ph-pill">● Model Ready</div>
</div>
""", unsafe_allow_html=True)

# ── Main logic ───────────────────────────────────────────────────────────────
if predict_btn:
    raw_input = {
        'age': age_val, 'sex': sex_val, 'cp': cp_val,
        'trestbps': trestbps_val, 'chol': chol_val, 'fbs': fbs_val,
        'restecg': restecg_val, 'thalach': thalach_val, 'exang': exang_val,
        'oldpeak': oldpeak_val, 'slope': slope_val, 'ca': ca_val, 'thal': thal_val,
    }

    enc_input  = get_raw_to_encoded(raw_input)
    pred_class = model.predict(enc_input.values)[0]
    pred_prob  = model.predict_proba(enc_input.values)[0][1]
    prob_pct   = round(pred_prob * 100, 1)

    importances = model.feature_importances_
    top3_idx    = np.argsort(importances)[::-1][:3]
    top3_feats  = [feat_names[i] for i in top3_idx]
    top3_vals   = importances[top3_idx]

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        sex_str = "Male" if sex_val == 1 else "Female"
        if pred_class == 1:
            st.markdown(f"""
            <div class="rc">
              <div class="rc-lbl">Prediction Outcome</div>
              <div class="rc-title-high">Disease Present</div>
              <div class="badge-h">⚠&nbsp; HIGH RISK</div>
              <div class="pb-wrap">
                <div class="pb-header">
                  <span class="pb-label">Disease Probability</span>
                  <span class="pb-pct-h">{prob_pct}%</span>
                </div>
                <div class="pb-track"><div class="pb-fill-h" style="width:{prob_pct}%"></div></div>
              </div>
              <div class="ss">
                <div class="sp"><div class="sp-val">{age_val}</div><div class="sp-lbl">Age</div></div>
                <div class="sp"><div class="sp-val">{sex_str[0]}</div><div class="sp-lbl">Sex</div></div>
                <div class="sp"><div class="sp-val">{thalach_val}</div><div class="sp-lbl">Max HR</div></div>
                <div class="sp"><div class="sp-val">{chol_val}</div><div class="sp-lbl">Chol.</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rc">
              <div class="rc-lbl">Prediction Outcome</div>
              <div class="rc-title-low">No Disease Detected</div>
              <div class="badge-l">✓&nbsp; LOW RISK</div>
              <div class="pb-wrap">
                <div class="pb-header">
                  <span class="pb-label">Disease Probability</span>
                  <span class="pb-pct-l">{prob_pct}%</span>
                </div>
                <div class="pb-track"><div class="pb-fill-l" style="width:{prob_pct}%"></div></div>
              </div>
              <div class="ss">
                <div class="sp"><div class="sp-val">{age_val}</div><div class="sp-lbl">Age</div></div>
                <div class="sp"><div class="sp-val">{sex_str[0]}</div><div class="sp-lbl">Sex</div></div>
                <div class="sp"><div class="sp-val">{thalach_val}</div><div class="sp-lbl">Max HR</div></div>
                <div class="sp"><div class="sp-val">{chol_val}</div><div class="sp-lbl">Chol.</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="cc-lbl">Top Predictive Features</div>', unsafe_allow_html=True)

            clean_labels = []
            for f in top3_feats[::-1]:
                name = f.replace('_', ' ')
                if ' ' in name and name.split()[-1].replace('.','').isdigit():
                    parts = name.rsplit(' ', 1)
                    val = parts[1].rstrip('0').rstrip('.')
                    name = f"{parts[0].title()} ({val})"
                else:
                    name = name.title()
                clean_labels.append(name)

            fig, ax = plt.subplots(figsize=(5.0, 2.8), facecolor='white')
            bar_colors = ['#BFDBFE', '#3B82F6', '#1E40AF']
            vals_rev = top3_vals[::-1]
            bars = ax.barh(clean_labels, vals_rev,
                           color=bar_colors, height=0.44, edgecolor='none')
            for bar, val in zip(bars, vals_rev):
                ax.text(bar.get_width() + max(vals_rev) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', ha='left',
                        fontsize=8.5, color='#64748B',
                        fontfamily='DejaVu Sans')
            ax.set_xlabel('Importance Score', fontsize=8, color='#94A3B8', labelpad=6)
            ax.tick_params(colors='#334155', labelsize=9.5, axis='y')
            ax.tick_params(colors='#94A3B8', labelsize=8, axis='x')
            ax.set_xlim(0, max(vals_rev) * 1.30)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color('#E2E8F0')
            ax.tick_params(axis='y', length=0)
            ax.set_facecolor('white')
            fig.patch.set_edgecolor('none')
            plt.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    top_feat = top3_feats[0].replace('_', ' ')
    if pred_class == 1:
        expl = (f"This patient presents with elevated cardiac risk markers. "
                f"The model's strongest predictor is <strong>{top_feat}</strong>, "
                f"alongside ST depression and fluoroscopy vessel count. "
                f"Cardiologist review and further diagnostic workup is recommended.")
        cls = "h"
    else:
        expl = ("This patient's clinical markers fall within low-risk parameters for coronary artery disease. "
                "Maximum heart rate achieved is adequate and ST depression is minimal. "
                "Routine follow-up at the next scheduled visit is advised.")
        cls = "l"

    st.markdown(f"""
    <div class="eb {cls}">
      <div class="eb-lbl">Clinical Interpretation</div>
      <div class="eb-txt">{expl}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

    with st.expander("View encoded feature vector sent to model"):
        st.dataframe(enc_input.T.rename(columns={0: 'Scaled Value'}),
                     use_container_width=True)

else:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""
        <div class="es">
          <div class="es-icon">🫀</div>
          <div class="es-title">Ready for Assessment</div>
          <div class="es-sub">
            The sidebar form is pre-populated with a real test patient.<br>
            Click <strong>Run Prediction</strong> to generate a risk analysis.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="ref-heading">Feature Reference</div>', unsafe_allow_html=True)
        ref_data = {
            "Feature":     ["age","sex","cp","trestbps","chol","fbs",
                            "restecg","thalach","exang","oldpeak","slope","ca","thal"],
            "Type":        ["Numeric","Binary","Categorical","Numeric","Numeric",
                            "Binary","Categorical","Numeric","Binary",
                            "Numeric","Categorical","Numeric","Categorical"],
            "Range":       ["20–80","0/1","1–4","80–200 mmHg","100–600 mg/dl",
                            "0/1","0–2","70–210 bpm","0/1","0.0–6.2",
                            "1–3","0–3","{3,6,7}"],
        }
        st.table(pd.DataFrame(ref_data))
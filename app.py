import streamlit as st
import numpy as np
import joblib
import pandas as pd
import base64
import mysql.connector

# ----------------------------
# Function to set background from local image
# ----------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white !important;
        font-weight: bold !important;
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    [data-testid="stSidebar"] {{
        background-color: rgba(0,0,0,0.7);
        color: white !important;
        font-weight: bold !important;
    }}
    .stMarkdown, .stText, .stNumberInput label, .stSelectbox label,
    .stRadio label, .stCheckbox label, .stTextInput label {{
        color: white !important;
        font-weight: bold !important;
    }}
    h1, h2, h3, h4, h5, h6, p, span, label {{
        color: white !important;
        font-weight: bold !important;
    }}
    .stAlert p {{
        color: white !important;
        font-weight: bold !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------------------
# Load model
# ----------------------------
best_model = joblib.load("./outputs/best_model_XGBoost.joblib")

# ----------------------------
# Set your local image path
# ----------------------------
add_bg_from_local("./images.jpg")

# ----------------------------
# Title
# ----------------------------
st.title("❤️ Heart Disease Prediction App")

# ----------------------------
# Input form
# ----------------------------
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])

# ----------------------------
# Preprocessing
# ----------------------------
sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0

input_dict = {
    "age": [age],
    "sex": [sex_val],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs_val],
    "thalch": [thalach],
    "exang": [exang_val],
    "oldpeak": [oldpeak],
    "cp": [cp],
    "restecg": [restecg],
    "slope": [slope]
}
df_input = pd.DataFrame(input_dict)

df_input = pd.get_dummies(df_input, columns=["cp", "restecg", "slope"])
model_features = [
    'age','sex','trestbps','chol','fbs','thalch','exang','oldpeak',
    'cp_asymptomatic','cp_atypical angina','cp_non-anginal','cp_typical angina',
    'restecg_lv hypertrophy','restecg_normal','restecg_st-t abnormality',
    'slope_downsloping','slope_flat','slope_upsloping'
]
df_input = df_input.reindex(columns=model_features, fill_value=0)

# ----------------------------
# Prediction + Save to MySQL
# ----------------------------
if st.button("Predict"):
    pred = best_model.predict(df_input)[0]
    proba = best_model.predict_proba(df_input)[0][1]

    result_text = ""
    if pred == 1:
        result_text = f"⚠️ Patient likely has Heart Disease (Risk Probability: {proba:.2f})"
        st.error(result_text)
    else:
        result_text = f"✅ Patient unlikely to have Heart Disease (Risk Probability: {proba:.2f})"
        st.success(result_text)

    # --- Save to MySQL ---
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="",         # 👉 your MySQL username
            password="",   # 👉 your MySQL password
            database="heart_disease_db"
        )
        cursor = conn.cursor()

        sql = """
        INSERT INTO predictions 
        (age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, cp, restecg, slope, prediction, probability)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (age, sex_val, trestbps, chol, fbs_val, thalach, exang_val, oldpeak, cp, restecg, slope, result_text, float(proba))
        cursor.execute(sql, values)
        conn.commit()

        st.success("✅ Data saved to MySQL database!")

    except Exception as e:
        st.error(f"Database Error: {e}")

    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

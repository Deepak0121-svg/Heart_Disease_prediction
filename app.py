from flask import Flask, render_template, request
import pandas as pd
import mysql.connector
import joblib
import os

app = Flask(__name__)

# ----------------------------
# Load Random Forest model
# ----------------------------
rf_model = joblib.load("./outputs/RandomForest_model.joblib")

# Expected feature order
model_features = [
    'age','sex','trestbps','chol','fbs','thalch','exang','oldpeak',
    'cp_asymptomatic','cp_atypical angina','cp_non-anginal','cp_typical angina',
    'restecg_lv hypertrophy','restecg_normal','restecg_st-t abnormality',
    'slope_downsloping','slope_flat','slope_upsloping'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ----------------------------
        # Collect inputs from form
        # ----------------------------
        age = int(request.form["age"])
        sex = request.form["sex"]        # Female / Male
        cp = request.form["cp"]          # chest pain type
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = request.form["fbs"]        # Yes / No
        restecg = request.form["restecg"]
        thalach = int(request.form["thalach"])
        exang = request.form["exang"]
        oldpeak = float(request.form["oldpeak"])
        slope = request.form["slope"]

        # ----------------------------
        # Encode categorical values
        # ----------------------------
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs == "Yes" else 0
        exang_val = 1 if exang == "Yes" else 0

        # ----------------------------
        # Prepare dataframe
        # ----------------------------
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

        # One-hot encode categorical
        df_input = pd.get_dummies(df_input, columns=["cp", "restecg", "slope"])
        df_input = df_input.reindex(columns=model_features, fill_value=0)

        # ----------------------------
        # Prediction
        # ----------------------------
        pred = rf_model.predict(df_input)[0]
        proba = rf_model.predict_proba(df_input)[0][1]

        if pred == 1:
            result_text = f"⚠️ Patient likely has Heart Disease (Risk Probability: {proba:.2f})"
        else:
            result_text = f"✅ Patient unlikely to have Heart Disease (Risk Probability: {proba:.2f})"

        # ----------------------------
        # Save to MySQL (optional)
        # ----------------------------
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="m936113@",  # your local MySQL password
                database="heart_disease_db",
                port=3306
            )
            cursor = conn.cursor()

            sql = """
            INSERT INTO predictions 
            (age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, cp, restecg, slope, prediction, probability)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (age, sex_val, trestbps, chol, fbs_val, thalach, exang_val,
                      oldpeak, cp, restecg, slope, result_text, float(proba))
            cursor.execute(sql, values)
            conn.commit()

        except mysql.connector.Error as db_err:
            result_text += f" (DB Error: {db_err})"

        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

        return render_template("index.html", prediction_text=result_text)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    # ✅ Important for Render: use PORT from environment, not default 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

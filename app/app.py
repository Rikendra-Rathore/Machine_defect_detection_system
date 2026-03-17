from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load trained pipeline
model_pipeline = joblib.load("../model/textile_defect_model.pkl")

# Extract model and preprocessor
model = model_pipeline.named_steps["model"]
preprocessor = model_pipeline.named_steps["preprocessor"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get user inputs
    temperature = float(request.form["temperature"])
    speed = float(request.form["speed"])
    torque = float(request.form["torque"])
    yarn_type = request.form["yarn_type"]

    # Create dataframe
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Speed": [speed],
        "Torque": [torque],
        "Yarn_Type": [yarn_type]
    })

    # Prediction
    prediction = model_pipeline.predict(input_data)[0]

    # Probability
    probability = model_pipeline.predict_proba(input_data)[0][1]

    # Health score
    health_score = 100 - probability * 100

    # Risk level
    if probability > 0.7:
        risk = "HIGH"
        color = "red"
    elif probability > 0.4:
        risk = "MEDIUM"
        color = "orange"
    else:
        risk = "LOW"
        color = "green"

    if prediction == 1:
        result = "Defect Detected"
    else:
        result = "Machine Working Normally"

    # -------- Feature Importance --------

    importances = model.feature_importances_

    # Get feature names from pipeline
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Create chart
    plt.figure(figsize=(8,4))
    plt.barh(feature_names, importances)

    plt.title("Feature Importance for Defect Prediction")
    plt.xlabel("Importance Score")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    importance_plot = base64.b64encode(buf.read()).decode("utf-8")

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2),
        health_score=round(health_score, 2),
        risk=risk,
        color=color,
        shap_plot=importance_plot
    )


@app.route("/multi_predict", methods=["POST"])
def multi_predict():

    file = request.files["file"]

    df = pd.read_csv(file)

    preds = model_pipeline.predict(df)

    df["Prediction"] = preds

    table = df.to_html(classes="table table-striped")

    return render_template(
        "index.html",
        table=table
    )


if __name__ == "__main__":
    app.run(debug=True)
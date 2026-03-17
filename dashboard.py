import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Textile Machine Health Monitoring",
    layout="wide"
)

st.title("🧵 AI Powered Textile Machine Health Monitoring System")

st.write(
"This system predicts machine defects and provides health insights for textile manufacturing."
)

# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("model/textile_defect_model.pkl")

# -----------------------------
# Sidebar Inputs
# -----------------------------

st.sidebar.header("Machine Input Parameters")

temperature = st.sidebar.slider("Temperature", 250, 400, 320)
speed = st.sidebar.slider("Machine Speed", 1000, 2000, 1400)
torque = st.sidebar.slider("Torque", 20, 80, 50)

yarn_type = st.sidebar.selectbox(
    "Yarn Type",
    ["Cotton", "Polyester", "Silk"]
)

# -----------------------------
# Create input dataframe
# -----------------------------

input_data = pd.DataFrame({
    "Temperature":[temperature],
    "Speed":[speed],
    "Torque":[torque],
    "Yarn_Type":[yarn_type]
})

# -----------------------------
# Show Machine Metrics
# -----------------------------

st.subheader("Machine Parameters")

col1, col2, col3 = st.columns(3)

col1.metric("Temperature", temperature)
col2.metric("Speed", speed)
col3.metric("Torque", torque)

# -----------------------------
# Show Input Data
# -----------------------------

st.subheader("Input Data Sent To Model")

st.write(input_data)

# -----------------------------
# Prediction Section
# -----------------------------

st.subheader("Defect Prediction")

if st.button("Predict Machine Status"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("Defect Probability:", round(probability * 100,2), "%")

    # Machine health score
    health_score = 100 - (probability * 100)

    st.metric("Machine Health Score", f"{round(health_score,2)} / 100")

    # Prediction result

    if prediction == 1:
        st.error("⚠ Defect Detected in Machine")
    else:
        st.success("✔ Machine Operating Normally")

    # Maintenance recommendation

    if probability > 0.7:
        st.warning("Recommended Action: Immediate machine inspection required")

    elif probability > 0.4:
        st.info("Recommended Action: Schedule maintenance soon")

    else:
        st.success("Machine is operating within safe parameters")

# -----------------------------
# Machine Monitoring Chart
# -----------------------------

st.subheader("Machine Parameter Monitoring")

chart_data = pd.DataFrame({
    "Temperature":[temperature],
    "Speed":[speed],
    "Torque":[torque]
})

st.bar_chart(chart_data)

# -----------------------------
# Batch Prediction Section
# -----------------------------

st.subheader("Batch Machine Prediction")

uploaded_file = st.file_uploader(
    "Upload Machine Data CSV",
    type=["csv"]
)

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data)

    data["Defect_Prediction"] = predictions

    st.write(data)

    st.download_button(
        label="Download Predictions",
        data=data.to_csv(index=False),
        file_name="machine_predictions.csv",
        mime="text/csv"
    )
import pandas as pd
import joblib

model = joblib.load("../model/textile_defect_model.pkl")

def predict_defect(data):

    df = pd.DataFrame(data)

    prediction = model.predict(df)

    return prediction
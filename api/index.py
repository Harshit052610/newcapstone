import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Security XGBoost API")

# Load model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "xgboost_model.pkl"
)

FEATURES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "xgboost_feature_columns.pkl"
)

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)


class SecurityLog(BaseModel):
    round_trip_time_ms: float = 0
    asn: int = 0
    hour: int = 0
    day_of_week: int = 0
    login_success: int = 0
    device_type: str = "Unknown"
    country: str = "Unknown"


@app.get("/")
def home():
    return {
        "status": "online",
        "model": "XGBoost",
        "features": len(feature_columns)
    }


@app.post("/predict")
def predict(log: SecurityLog):

    data = pd.DataFrame([{
        "Round-Trip Time [ms]": log.round_trip_time_ms,
        "ASN": log.asn,
        "hour": log.hour,
        "day_of_week": log.day_of_week,
        "login_success": log.login_success,
        "Device Type": log.device_type,
        "Country": log.country
    }])

    # Same one-hot encoding used during training
    data = pd.get_dummies(
        data,
        columns=["Device Type", "Country"],
        drop_first=True
    )

    # Make sure inference has exactly the same 208 columns
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    data = data.replace([float("inf"), float("-inf")], 0)
    data = data.fillna(0)

    prediction = int(model.predict(data)[0])

    probability = float(
        model.predict_proba(data)[0][1]
    )

    return {
        "prediction": "ATTACK" if prediction == 1 else "NORMAL",
        "prediction_value": prediction,
        "confidence": round(probability, 4),
        "model": "XGBoost"
    }

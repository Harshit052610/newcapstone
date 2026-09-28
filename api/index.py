import json
import os
import xgboost as xgb
import numpy as np
from http.server import BaseHTTPRequestHandler


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "xgboost_model.json"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "xgboost_input_features.json"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = xgb.Booster()
model.load_model(MODEL_PATH)


# --------------------------------------------------
# LOAD FEATURE COLUMNS
# --------------------------------------------------

with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)


# --------------------------------------------------
# FEATURE CREATION
# --------------------------------------------------

def make_features(data):

    features = np.zeros(
        (1, len(feature_columns)),
        dtype=np.float32
    )

    # Numerical features
    values = {
        "Round-Trip Time [ms]": data.get(
            "round_trip_time_ms",
            0
        ),

        "ASN": data.get(
            "asn",
            0
        ),

        "hour": data.get(
            "hour",
            0
        ),

        "day_of_week": data.get(
            "day_of_week",
            0
        ),

        "login_success": data.get(
            "login_success",
            0
        )
    }

    for name, value in values.items():

        if name in feature_columns:

            features[
                0,
                feature_columns.index(name)
            ] = float(value)


    # Device Type
    device = str(
        data.get(
            "device_type",
            "unknown"
        )
    ).lower()

    device_column = f"Device Type_{device}"

    if device_column in feature_columns:

        features[
            0,
            feature_columns.index(device_column)
        ] = 1


    # Country
    country = str(
        data.get(
            "country",
            "unknown"
        )
    ).upper()

    country_column = f"Country_{country}"

    if country_column in feature_columns:

        features[
            0,
            feature_columns.index(country_column)
        ] = 1


    return features


# --------------------------------------------------
# VERCEL HANDLER
# --------------------------------------------------

class handler(BaseHTTPRequestHandler):


    # --------------------------------------------------
    # SEND JSON RESPONSE
    # --------------------------------------------------

    def send_json(self, status, data):

        response = json.dumps(
            data
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(response))
        )

        self.end_headers()

        self.wfile.write(response)


    # --------------------------------------------------
    # GET
    # --------------------------------------------------

    def do_GET(self):

        self.send_json(
            200,
            {
                "status": "online",
                "model": "XGBoost",
                "features": len(feature_columns)
            }
        )


    # --------------------------------------------------
    # POST
    # --------------------------------------------------

    def do_POST(self):

        try:

            # Read request body
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
            )

            # Parse JSON
            data = json.loads(body)


            # Create 208-feature vector
            features = make_features(data)


            # Create XGBoost input
            dmatrix = xgb.DMatrix(
                features
            )


            # Get attack probability
            attack_probability = float(
                model.predict(
                    dmatrix
                )[0]
            )


            # Classification
            prediction = (
                1
                if attack_probability >= 0.3
                else 0
            )


            # Confidence of predicted class
            confidence = (
                attack_probability
                if prediction == 1
                else 1 - attack_probability
            )


            # Return result
            self.send_json(
                200,
                {
                    "prediction":
                        "ATTACK"
                        if prediction == 1
                        else "NORMAL",

                    "prediction_value":
                        prediction,

                    "attack_probability":
                        round(
                            attack_probability,
                            4
                        ),

                    "confidence":
                        round(
                            confidence,
                            4
                        ),

                    "model":
                        "XGBoost"
                }
            )


        except Exception as e:

            self.send_json(
                500,
                {
                    "error": str(e)
                }
            )

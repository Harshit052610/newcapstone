import json
import os
import xgboost as xgb
import numpy as np
from http.server import BaseHTTPRequestHandler


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "xgboost_model.json"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "xgboost_input_features.json"
)


# Load native XGBoost model
model = xgb.Booster()
model.load_model(MODEL_PATH)


# Load the exact 208 feature names
with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)


def make_features(data):

    features = np.zeros(
        (1, len(feature_columns)),
        dtype=np.float32
    )

    # Numerical features
    values = {
        "Round-Trip Time [ms]": data.get(
            "round_trip_time_ms", 0
        ),
        "ASN": data.get("asn", 0),
        "hour": data.get("hour", 0),
        "day_of_week": data.get("day_of_week", 0),
        "login_success": data.get(
            "login_success", 0
        ),
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

    device_column = (
        f"Device Type_{device}"
    )

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

    country_column = (
        f"Country_{country}"
    )

    if country_column in feature_columns:

        features[
            0,
            feature_columns.index(country_column)
        ] = 1


    return features


class handler(BaseHTTPRequestHandler):

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


    def do_GET(self):

        self.send_json(
            200,
            {
                "status": "online",
                "model": "XGBoost",
                "features": len(
                    feature_columns
                )
            }
        )


    def do_POST(self):

    try:

        content_length = int(
            self.headers.get(
                "Content-Length",
                0
            )
        )

        body = self.rfile.read(
            content_length
        )

        data = json.loads(body)

        features = make_features(data)

        dmatrix = xgb.DMatrix(features)

        probability = float(
            model.predict(dmatrix)[0]
        )

        prediction = (
            1
            if probability >= 0.5
            else 0
        )

        self.send_json(
            200,
            {
                "prediction":
                    "ATTACK"
                    if prediction == 1
                    else "NORMAL",

                "prediction_value":
                    prediction,

                "confidence":
                    round(probability, 4),

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


        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
            )

            data = json.loads(body)

            features = make_features(
                data
            )


            # Native XGBoost prediction
dmatrix = xgb.DMatrix(features)

probability = float(
    model.predict(dmatrix)[0]
)

prediction = (
    1
    if probability >= 0.5
    else 0
)


            self.send_json(
                200,
                {
                    "prediction":
                        "ATTACK"
                        if prediction == 1
                        else "NORMAL",

                    "prediction_value":
                        prediction,

                    "confidence":
                        round(
                            probability,
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

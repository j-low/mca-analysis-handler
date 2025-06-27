# app.py

import os
import pandas as pd
from flask import Flask, request, jsonify

from data_preparation_pipeline import DataPreparationPipeline
from prediction_utils import load_prediction_model, make_predictions

# ------------------------------------------------------------------------------
# Load secrets from environment (Render will inject these)
# ------------------------------------------------------------------------------
BLS_API_KEY = os.getenv("BLS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
BEA_API_KEY = os.getenv("BEA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------------------
# Initialize pipeline & model once at startup
# ------------------------------------------------------------------------------
pipeline = DataPreparationPipeline(
    bea_api_key=BEA_API_KEY,
    fred_api_key=FRED_API_KEY,
    bls_api_key=BLS_API_KEY,
    openai_api_key=OPENAI_API_KEY,
)
model = load_prediction_model("default_probability_model")

# ------------------------------------------------------------------------------
# Create Flask app
# ------------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts a JSON array of flat objects representing MCA deals.
    Returns a JSON array of enriched + predicted results.
    """
    try:
        payload = request.get_json()
        if not isinstance(payload, list):
            return jsonify(error="Request body must be a JSON array of objects"), 400

        # 1. Load into DataFrame
        df = pd.DataFrame(payload)

        # 2. Normalize & enrich
        prepared = pipeline.prepare_data_for_prediction(df)

        # 3. Predict
        results_df, _ = make_predictions(model, prepared)

        # 4. Serialize to JSON
        records = results_df.to_dict(orient="records")
        return jsonify(results=records), 200

    except Exception as e:
        # Return human‐readable error
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    # Port can be overridden by Render via PORT env var
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

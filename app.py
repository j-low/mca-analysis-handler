# app.py

import os
import pandas as pd
from flask import Flask, request, jsonify
import logging
from typing import cast

from data_preparation_pipeline import DataPreparationPipeline
from prediction_utils import load_prediction_model, make_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Load secrets from environment
# ----------------------------------------------------------------------------
BLS_API_KEY = os.getenv("BLS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
BEA_API_KEY = os.getenv("BEA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not all([BLS_API_KEY, FRED_API_KEY, BEA_API_KEY, OPENAI_API_KEY]):
    raise RuntimeError("Missing required API keys")

# Cast to str
BLS_API_KEY = cast(str, BLS_API_KEY)
FRED_API_KEY = cast(str, FRED_API_KEY)
BEA_API_KEY = cast(str, BEA_API_KEY)
OPENAI_API_KEY = cast(str, OPENAI_API_KEY)

# ----------------------------------------------------------------------------
# Initialize pipeline & model
# ----------------------------------------------------------------------------
pipeline = DataPreparationPipeline(
    bea_api_key=BEA_API_KEY,
    fred_api_key=FRED_API_KEY,
    bls_api_key=BLS_API_KEY,
    openai_api_key=OPENAI_API_KEY,
)
cpi = pipeline.fetch_cpi_data()
nat = pipeline.fetch_national_unemployment_data()
state = pipeline.fetch_state_unemployment_data()
logger.info("DATA FETCHED")
logger.info(f"CPI columns: {cpi.columns.tolist()}")
logger.info(f"National unemployment columns: {nat.columns.tolist()}")
logger.info(f"State unemployment columns: {state.columns.tolist()}")
model = load_prediction_model("default_probability_model")

# ----------------------------------------------------------------------------
# Create Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "Service is running"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        payload = request.get_json()
        logger.info(f"Received payload with {len(payload)} records")
        if not isinstance(payload, list):
            logger.error("Invalid payload format - expected JSON array")
            return jsonify(error="Request body must be a JSON array of objects"), 400

        # 1. Load into DataFrame
        df = pd.DataFrame(payload)
        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"Input columns: {df.columns.tolist()}")

        # 2. Clean raw inputs
        df = pipeline.clean_data(df)
        logger.info(f"Cleaned DataFrame dtypes:\n{df.dtypes}")

        # 3. Normalize & enrich
        prepared = pipeline.prepare_data_for_prediction(df)
        logger.info(f"Prepared DataFrame shape: {prepared.shape}")
        logger.info(f"Prepared columns: {prepared.columns.tolist()}")

        # 4. Predict
        results_df, pred_counts = make_predictions(model, prepared)
        logger.info(f"Prediction counts: {pred_counts}")

        # 5. Replace any NaN (or None) with None so JSON is valid
        results_df = results_df.where(pd.notnull(results_df), None)

        # 6. Serialize to JSON
        records = results_df.to_dict(orient="records")
        logger.info(f"Returning {len(records)} results")
        return jsonify(records), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)

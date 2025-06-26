import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd

from data_preparation_pipeline import DataPreparationPipeline
from prediction_utils import load_prediction_model, make_predictions

# Load .env (or use Render’s env vars)
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "default_probability_model")

# Initialize pipeline & model once
pipeline = DataPreparationPipeline()
try:
    model = load_prediction_model(model_path=MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load prediction model: {e}")

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST JSON for one MCA deal; returns a JSON object containing:
    - All normalized fields from the pipeline
    - Any extra quantitative metrics the pipeline added (e.g. grouped_status)
    - predicted_status
    """
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        # 1) Data preparation
        df = pd.DataFrame([payload])
        prepared_df = pipeline.prepare_data_for_prediction(df)
        if prepared_df is None or prepared_df.empty:
            return jsonify({"error": "Data preparation failed or returned no rows"}), 400

        # 2) Prediction
        results_df, _ = make_predictions(model, prepared_df)

        # 3) Extract row as dict
        result_row = results_df.iloc[0].to_dict()

        # 4) Return all columns
        return jsonify(result_row)

    except Exception as e:
        # You can expand this to log to a file or external logger
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

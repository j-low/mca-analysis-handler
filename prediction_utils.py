# prediction_utils.py

import pandas as pd
import pycaret.classification as clf
from typing import Tuple, Dict

def load_prediction_model(model_path: str = 'default_probability_model') -> object:
    """
    Load the saved PyCaret classification model.
    """
    try:
        model = clf.load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def make_predictions(model: object, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Run model.predict on data (drops 'balance' column before).
    """
    try:
        preds = model.predict(data.drop(columns=['balance']))
        results_df = data.copy()
        results_df['predicted_status'] = preds
        pred_counts = results_df['predicted_status'].value_counts().to_dict()
        return results_df, pred_counts
    except Exception as e:
        raise Exception(f"Failed to make predictions: {str(e)}")

def get_prediction_metrics(pred_counts: Dict[str,int]) -> Dict[str,int]:
    """
    Return count of default vs paid-off.
    """
    return {
        'default':    pred_counts.get('default', 0),
        'paid_off':   pred_counts.get('paid-off', 0)
    }

# prediction_utils.py

import pandas as pd
import pycaret.classification as clf
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

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
        # Log shape and columns for debugging
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input columns: {data.columns.tolist()}")
        
        # Keep only the features used during training
        feature_columns = [
            'total_funding', 'factor_rate', 'term', 'advance_number', 
            'holdback_percent', 'fico_score', 'business_age', 'position',
            'industry_category', 'grouped_pay_type', 'cpi', 
            'yoy_percentage_change', 'unemployment_rate', 'state_unemployment_rate',
            'tier', 'type'  # Added type as it's also required by the model
        ]
        
        # Make prediction using only the required features
        pred_data = data[feature_columns]
        logger.info(f"Prediction data shape: {pred_data.shape}")
        logger.info(f"Prediction columns: {pred_data.columns.tolist()}")
        
        # Handle different prediction return types
        preds = model.predict(pred_data)
        if isinstance(preds, list):
            preds = pd.Series(preds, index=data.index)
        
        # Add predictions and count
        results_df = data.copy()
        results_df['predicted_status'] = preds
        pred_counts = results_df['predicted_status'].value_counts().to_dict()
        
        logger.info(f"Prediction counts: {pred_counts}")
        return results_df, pred_counts
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise Exception(f"Failed to make predictions: {str(e)}")

def get_prediction_metrics(pred_counts: Dict[str,int]) -> Dict[str,int]:
    """
    Return count of default vs paid-off.
    """
    return {
        'default':    pred_counts.get('default', 0),
        'paid_off':   pred_counts.get('paid-off', 0)
    }

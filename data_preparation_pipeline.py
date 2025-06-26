# data_preparation_pipeline.py

import os
import pandas as pd
import numpy as np
import requests as r
import logging
from datetime import datetime
from openai import OpenAI
import tempfile
import gdown
import json
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataPreparationPipeline:
    """Data preparation pipeline for MCA prediction model"""

    def __init__(self):
        # Pull secrets from environment (Render injects these)
        self.bea_api_key    = os.getenv("BEA_API_KEY")
        self.fred_api_key   = os.getenv("FRED_API_KEY")
        self.bls_api_key    = os.getenv("BLS_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        missing = [k for k in ("BEA_API_KEY","FRED_API_KEY","BLS_API_KEY") if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

        # State mapping…
        self.us_state_abbrev = { … }    # unchanged
        self.state_replacements = { … } # unchanged

    # … group_status, group_pay_type, clean_column_name unchanged …

    def fetch_cpi_data(self):
        """Fetch CPI data from FRED"""
        series_id = "MEDCPIM158SFRBCLE"
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_api_key}&file_type=json"
        resp = r.get(url)
        # … unchanged logic …

    def fetch_state_unemployment_data(self):
        """Fetch state unemployment via BLS"""
        # use self.bls_api_key in payload
        # … unchanged logic …

    def fetch_national_unemployment_data(self):
        """Fetch national unemployment from FRED"""
        series_id = "UNRATE"
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_api_key}&file_type=json"
        resp = r.get(url)
        # … unchanged logic …

    def fetch_industry_gdp_data(self):
        """Fetch industry GDP from BEA"""
        url = (
            f"https://apps.bea.gov/api/data/?&UserID={self.bea_api_key}"
            "&method=GetData&DataSetName=GDPbyIndustry&Year=ALL&Industry=ALL&tableID=1&Frequency=A"
        )
        resp = r.get(url)
        # … unchanged logic …

    def prepare_data_for_prediction(self, export_df: pd.DataFrame) -> pd.DataFrame:
        """Main entrypoint"""
        logger.info("Preparing data for prediction…")
        df = export_df.copy()
        # … the rest of your Step 1–3 pipeline unchanged …
        return df

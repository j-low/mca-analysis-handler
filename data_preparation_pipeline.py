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

    def __init__(self, bea_api_key: str, fred_api_key: str, bls_api_key: str, openai_api_key: str):
        # assign from caller rather than re-reading os.getenv()
        self.bea_api_key    = bea_api_key
        self.fred_api_key   = fred_api_key
        self.bls_api_key    = bls_api_key
        self.openai_api_key = openai_api_key

        # validate presence
        missing = [name for name, val in [
            ("BEA_API_KEY", bea_api_key),
            ("FRED_API_KEY", fred_api_key),
            ("BLS_API_KEY", bls_api_key),
            ("OPENAI_API_KEY", openai_api_key),
        ] if not val]
        if missing:
            raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

        # State abbreviation mapping
        self.us_state_abbrev = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
            'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
            'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
            'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
            'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
            'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
            'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
            'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
            'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
            'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
            'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 'Puerto Rico': 'PR'
        }
        # State replacements for cleaning
        self.state_replacements = {
            'ARKANSAS': 'AR', 'BOOKLYN': 'NY', 'NY ': 'NY', 'FL ': 'FL',
            'TX ': 'TX', 'NJ ': 'NJ', 'NV ': 'NV', 'Il': 'IL', 'Ny': 'NY', 'Fl': 'FL'
        }

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

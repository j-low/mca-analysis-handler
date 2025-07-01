# data_preparation_pipeline.py

import os
import pandas as pd
import numpy as np
import requests as r
import logging
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class DataPreparationPipeline:
    """Data preparation pipeline for MCA prediction model"""

    def __init__(self,
                 bea_api_key: str,
                 fred_api_key: str,
                 bls_api_key: str,
                 openai_api_key: str):
        # assign keys
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

    def fetch_cpi_data(self) -> pd.DataFrame:
        """Fetch CPI data from FRED and compute YoY percentage change"""
        series_id = "MEDCPIM158SFRBCLE"
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={self.fred_api_key}&file_type=json"
        )
        resp = r.get(url)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"value": "cpi"}).set_index("date")
        df["cpi"] = df["cpi"].astype(float)
        df["yoy_percentage_change"] = df["cpi"].pct_change(12) * 100

        logger.info(f"[CPI] head:\n{df.head()}\ncolumns: {df.columns.tolist()}")
        return df

    def fetch_national_unemployment_data(self) -> pd.DataFrame:
        """Fetch national unemployment data from FRED"""
        series_id = "UNRATE"
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={self.fred_api_key}&file_type=json"
        )
        resp = r.get(url)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"value": "unemployment_rate"}).set_index("date")
        df["unemployment_rate"] = df["unemployment_rate"].astype(float)

        logger.info(f"[NAT_UNEMP] head:\n{df.head()}\ncolumns: {df.columns.tolist()}")
        return df

    def fetch_state_unemployment_data(self) -> pd.DataFrame:
        """Fetch state-level unemployment data from BLS"""
        # build series IDs from the two-letter state codes
        series_ids = [
            f"LAUST{abbr}0000000000003"
            for abbr in self.us_state_abbrev.values()
        ]
        payload = {
            "seriesid": series_ids,
            "startyear": "2010",
            "endyear": str(datetime.today().year),
            "registrationKey": self.bls_api_key
        }
        resp = r.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", json=payload)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for series in data.get("Results", {}).get("series", []):
            state_code = series["seriesID"][4:6]
            for item in series.get("data", []):
                # BLS returns periods like "M01".."M12" under periodName; map to month
                month = datetime.strptime(item["periodName"], "%B").month if item.get("periodName") else None
                # Build date only if year and month are valid
                if month is not None and item.get("year"):
                    records.append({
                        "date": pd.to_datetime(f"{item['year']}-{month:02d}-01"),
                        "state": state_code,
                        "state_unemployment_rate": float(item.get("value", 0))
                    })

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No state unemployment data returned; returning empty DataFrame")
            return df

        df = df.set_index("date")
        logger.info(f"[STATE_UNEMP] head:\n{df.head()}\ncolumns: {df.columns.tolist()}")
        return df

    def fetch_industry_gdp_data(self) -> pd.DataFrame:
        """Fetch industry GDP data from BEA"""
        url = (
            f"https://apps.bea.gov/api/data/"
            f"?&UserID={self.bea_api_key}"
            f"&method=GetData&DataSetName=GDPbyIndustry&Year=ALL&Industry=ALL&tableID=1&Frequency=A"
        )
        resp = r.get(url)
        resp.raise_for_status()

        # --- parse JSON and unwrap any list wrappings ---
        data = resp.json()
        logger.info(f"BEA API raw response: {data}")

        # Top-level might be a list: grab its first element
        if isinstance(data, list):
            if not data:
                logger.warning("Empty BEA response; returning empty DataFrame")
                return pd.DataFrame()
            data = data[0]

        # If it's not a dict at this point, bail
        if not isinstance(data, dict):
            logger.warning("Unexpected BEA response format; returning empty DataFrame")
            return pd.DataFrame()

        # --- extract the BEAAPI section, which itself might be a list ---
        bea = data.get("BEAAPI", {})
        if isinstance(bea, list):
            if not bea:
                logger.warning("Empty BEAAPI list; returning empty DataFrame")
                return pd.DataFrame()
            bea = bea[0]

        if not isinstance(bea, dict):
            logger.warning("Unexpected BEAAPI format; returning empty DataFrame")
            return pd.DataFrame()

        # --- dig into Results → Data ---
        results = bea.get("Results", {})
        if isinstance(results, list):
            if not results:
                logger.warning("Empty Results list; returning empty DataFrame")
                return pd.DataFrame()
            results = results[0]

        records = results.get("Data", []) if isinstance(results, dict) else []
        if not records:
            logger.warning("No industry GDP data returned; returning empty DataFrame")
            return pd.DataFrame()

        # --- build the DataFrame ---
        df = pd.DataFrame(records)
        # Convert Year to integer for merging
        df["Year"] = df["Year"].astype(int)
        df = df.rename(columns={"DataValue": "industry_gdp", "Year": "date"})
        df["industry_gdp"] = df["industry_gdp"].astype(float)
        df = df.set_index("date")

        logger.info(f"[GDP] head:\n{df.head()}\ncolumns: {df.columns.tolist()}")
        return df

    def _group_industry(self, industry: str) -> str:
        """Map raw industry values to broader categories"""
        # These mappings should match what the model was trained on
        industry = str(industry).lower()
        if 'restaurant' in industry or 'food' in industry:
            return 'Food & Restaurant'
        elif 'retail' in industry or 'shop' in industry or 'store' in industry:
            return 'Retail'
        elif 'service' in industry:
            return 'Services'
        elif 'construction' in industry or 'contractor' in industry:
            return 'Construction'
        elif 'medical' in industry or 'health' in industry:
            return 'Healthcare'
        elif 'transport' in industry or 'logistics' in industry:
            return 'Transportation'
        else:
            return 'Other'

    def _group_pay_type(self, pay_type: str) -> str:
        """Group payment types into broader categories"""
        # These mappings should match what the model was trained on
        pay_type = str(pay_type).lower()
        if 'ach' in pay_type:
            return 'ACH'
        elif 'split' in pay_type:
            return 'Split'
        elif 'lockbox' in pay_type:
            return 'Lockbox'
        else:
            return 'Other'

    def _prepare_gdp_data(self, gdp_data: dict) -> pd.DataFrame:
        """Process GDP data into a DataFrame with proper date index"""
        gdp_records = []
        for record in gdp_data['Results']['Data']:
            gdp_records.append({
                'date': record['Year'],
                'TableID': record['TableID'],
                'Frequency': record['Frequency'],
                'Quarter': record['Quarter'],
                'Industry': record['Industry'],
                'IndustrYDescription': record['IndustrYDescription'],
                'industry_gdp': float(record['DataValue']),
                'NoteRef': record['NoteRef']
            })
        
        df = pd.DataFrame(gdp_records)
        df['date'] = pd.to_datetime(df['date'], format='%Y')
        df.set_index('date', inplace=True)
        return df

    def prepare_data_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for making predictions by adding required features"""
        # Convert funding_date to datetime
        df['funding_date'] = pd.to_datetime(df['funding_date'])
        df['funding_year'] = df['funding_date'].dt.year
        
        # Add industry category and grouped pay type
        df['industry_category'] = df['industry'].apply(self._group_industry)
        df['grouped_pay_type'] = df['pay_type'].apply(self._group_pay_type)
        
        # Add CPI data
        df = df.merge(
            self.fetch_cpi_data()[["cpi", "yoy_percentage_change"]],
            left_on="funding_date", right_index=True,
            how="left"
        )
        
        # Add national unemployment data
        df = df.merge(
            self.fetch_national_unemployment_data(),
            left_on="funding_date", right_index=True,
            how="left"
        )
        
        # Add state unemployment data if available
        state_unemp = self.fetch_state_unemployment_data()
        if not state_unemp.empty:
            df = df.merge(
                state_unemp,
                left_on=["funding_date", "state"], right_index=True,
                how="left"
            )
        else:
            df['state_unemployment_rate'] = None
            logger.warning("No state unemployment data available - using null values")

        # Add GDP data - merge on funding_year
        logger.info(f"DataFrame before GDP merge - funding_year type: {df['funding_year'].dtype}")
        gdp_data = self.fetch_industry_gdp_data().copy()
        gdp_data.index = pd.to_datetime(gdp_data.index)
        logger.info(f"GDP index type: {gdp_data.index.dtype}")
        
        # Convert funding_year to datetime for merging
        df['merge_year'] = pd.to_datetime(df['funding_year'].astype(str), format='%Y')
        df = df.merge(
            gdp_data[['industry_gdp']],
            left_on='merge_year',
            right_index=True,
            how='left'
        )
        df.drop('merge_year', axis=1, inplace=True)

        logger.info(f"[MERGED DF] columns: {df.columns.tolist()}")
        logger.info(f"[MERGED DF] head:\n{df.head()}")
        
        return df


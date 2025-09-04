"""Data exploration and preprocessing for Beijing PM2.5 dataset."""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class BeijingPM25Explorer:
    """
    Data exploration and preprocessing for Beijing PM2.5 dataset.
    """

    def __init__(self, data_path: str):
        """Initialization of the dataset path, dataframe, and missing summary."""

        # sanity checks: data path must be valid
        if data_path is None:
            raise ValueError("data_path cannot be None")
        if not data_path or not data_path.strip():
            raise ValueError("data_path cannot be empty or whitespace")

        # initialisation
        self.data_path: str = data_path
        self.df: pd.DataFrame | None = None
        self.missing_summary: pd.DataFrame | None = None

    def load_and_inspect(self):
        """Load data and perform initial inspection"""

        print("Loading Beijing PM2.5 dataset...")
        self.df = pd.read_csv(self.data_path)

        print(f"Dataset shape: {self.df.shape}")
        print(
            f"Date range: {self.df['year'].min()}-{self.df['month'].min():02d} to {self.df['year'].max()}-{self.df['month'].max():02d}"
        )

        # Display basic info
        print("\nColumn names and types:")
        print(self.df.dtypes)

        print("\nFirst few rows:")
        print(self.df.head())

        return self.df

    def _create_datetime_features(self):
        """
        Cyclical feature encoding and datetime index creation:
        -> Enable time-based pandas operations (rolling, ...).
        -> Project onto orthogonal basis of cos/sin [-1, 1]-codomain, harmonising the data based on per-variable periodicity.
        -> Recover continuous features, addressing the discontinuities of conventional time keeping.

        => harmonise all data by transforming onto a unit circle.
        """
        # Create datetime from components
        self.df["datetime"] = pd.to_datetime(self.df[["year", "month", "day", "hour"]])

        # Create cyclical features to handle discontinuities
        # Hour: 0-23 -> continuous circular representation
        self.df["hour_sin"] = np.sin(2 * np.pi * self.df["hour"] / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * self.df["hour"] / 24)

        # Day of year: 1-365/366 -> continuous circular
        day_of_year = self.df["datetime"].dt.dayofyear
        self.df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        self.df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Month: 1-12 -> continuous circular
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)

        # Day of week
        dow = self.df["datetime"].dt.dayofweek
        self.df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        self.df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Set datetime as index for time-based pandas functionalities
        self.df.set_index("datetime", inplace=True)

        print("Created cyclical temporal features:")
        cyclical_cols = [
            col for col in self.df.columns if "_sin" in col or "_cos" in col
        ]
        print(f"  {cyclical_cols}")

        return self.df

    def _create_wind_features(self, drop_original: bool = False):
        """
        Encodes the categorical wind direction feature (cbwd) into a continuous,
        cyclical representation, correctly handling the 'calm' state.

        -> Creates a binary feature `is_calm`.
        -> Maps directional categories to degrees and projects to a (sin, cos) vector.
        -> Represents the 'calm' state as a zero vector (0, 0)---a unique point in the (cos, sin) space.

        => Preserves a metric, so that eg NW and N are close in the (cos, sin) space.

        Params
        ------
        drop_original: bool (default: False)
            Whether to drop the original categorical columns after encoding.
        """
        # early exit if not needed
        if "cbwd" not in self.df.columns:
            print("Wind direction column 'cbwd' not found. Skipping encoding.")
            return self.df

        # 1. Handle the 'calm' variable separately via one-hot encoding
        self.df["is_calm"] = (self.df["cbwd"] == "cv").astype(int)

        # 2. Map the known directions to degrees
        direction_mapping = {
            "N": 0.0,
            "NNE": 22.5,
            "NE": 45.0,
            "ENE": 67.5,
            "E": 90.0,
            "ESE": 112.5,
            "SE": 135.0,
            "SSE": 157.5,
            "S": 180.0,
            "SSW": 202.5,
            "SW": 225.0,
            "WSW": 247.5,
            "W": 270.0,
            "WNW": 292.5,
            "NW": 315.0,
            "NNW": 337.5,
        }

        self.df["cbwd_deg"] = self.df["cbwd"].map(direction_mapping)

        # 3. Apply cyclical encoding with a 360-degree period
        # The 'cv' entries will result in NaN, which is intentional
        self.df["cbwd_sin"] = np.sin(2 * np.pi * self.df["cbwd_deg"] / 360)
        self.df["cbwd_cos"] = np.cos(2 * np.pi * self.df["cbwd_deg"] / 360)

        # 4. Fill the NaN values (from 'cv' entries) with zero
        # This creates the (0, 0) vector for calm wind, representing no direction.
        self.df.loc[self.df["is_calm"] == 1, ["cbwd_sin", "cbwd_cos"]] = (
            0  # unique point in (cos, sin) space
        )

        # 5. Drop the intermediate and original categorical columns
        if drop_original:
            self.df.drop(columns=["cbwd", "cbwd_deg"], inplace=True)

        print("Created cyclical wind features:")
        print(f"  ['is_calm', 'cbwd_sin', 'cbwd_cos']")

        return self.df

    def feature_transform(self):
        """Main feature transformation pipeline, effectively mapping all relevant feature to a unit circle."""
        self._create_datetime_features()
        self._create_wind_features()
        return self.df

    def analyze_missing_data(self):
        """Comprehensive missing data analysis."""
        print("\n" + "=" * 50)
        print("MISSING DATA ANALYSIS")
        print("=" * 50)

        # Overall missing data summary
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100

        self.missing_summary = pd.DataFrame(
            {"Missing_Count": missing_counts, "Missing_Percentage": missing_pct}
        ).sort_values("Missing_Percentage", ascending=False)

        print("Missing data summary:")
        print(self.missing_summary[self.missing_summary["Missing_Count"] > 0])

        # PM2.5 specific missing pattern analysis
        if "pm2.5" in self.df.columns:
            pm25_missing = self.df["pm2.5"].isnull()
            print(
                f"\nPM2.5 missing data: {pm25_missing.sum()} samples ({pm25_missing.mean()*100:.1f}%)"
            )

            # Find consecutive missing periods
            missing_runs = self._find_consecutive_missing("pm2.5")
            if missing_runs:
                print(f"Longest consecutive missing period: {max(missing_runs)} hours")
                print(
                    f"Number of missing periods > 24h: {sum(1 for x in missing_runs if x > 24)}"
                )

        return self.missing_summary

    def _find_consecutive_missing(self, column):
        """Find consecutive missing value runs"""
        is_missing = self.df[column].isnull()
        runs = []
        current_run = 0

        for missing in is_missing:
            if missing:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        if current_run > 0:
            runs.append(current_run)

        return runs

    def get_data_summary(self):
        """Return comprehensive data summary"""
        summary = {
            "shape": self.df.shape,
            "date_range": (
                (self.df.index.min(), self.df.index.max())
                if self.df.index.name == "datetime"
                else None
            ),
            "missing_summary": self.missing_summary,
            "numeric_columns": self.df.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "pm25_stats": (
                self.df["pm2.5"].describe() if "pm2.5" in self.df.columns else None
            ),
        }
        return summary

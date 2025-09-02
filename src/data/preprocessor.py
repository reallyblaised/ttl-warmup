"""Data exploration and preprocessing for Beijing PM2.5 dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
import scienceplots

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

    def generate_eda_plots(self, output_dir: str = "plots/eda"):
        """
        Generates and saves a series of professional-quality EDA plots.
        Orchestrates all plotting sub-routines.

        Args:
            output_dir (str): The directory to save the plots.
        """
        # Ensure the output directory exists
        plots_path = Path(output_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        # Use a professional plot style from SciencePlots for the entire block
        with plt.style.context(["science", "ieee"]):

            # Call the individual plotting functions
            self._plot_missing_patterns(plots_path)
            self._plot_basic_distributions(plots_path)
            self._plot_temporal_patterns(plots_path)

        print(f"\nAll EDA plots saved as PDFs in the '{output_dir}' directory.")

    def _plot_missing_patterns(self, plots_path: Path):
        """Visualizes missing data patterns and saves to PDF."""
        print("Generating missing data patterns plot...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1.1. Missing data heatmap
        missing_data = self.df.isnull()
        key_cols_missing = ["pm2.5", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        available_cols_missing = [
            col for col in key_cols_missing if col in self.df.columns
        ]
        if available_cols_missing:
            sns.heatmap(
                missing_data[available_cols_missing].head(1000),
                yticklabels=False,
                cbar=True,
                ax=axes[0, 0],
                cmap="viridis",
            )
            axes[0, 0].set_title("Missing Data Patterns (First 1000 samples)")

        # 1.2. PM2.5 availability over time
        if "pm2.5" in self.df.columns:
            monthly_missing = (
                self.df["pm2.5"].resample("M").apply(lambda x: x.isnull().sum())
            )
            monthly_missing.plot(kind="bar", ax=axes[0, 1])
            axes[0, 1].set_title("PM2.5 Missing Data by Month")
            axes[0, 1].set_xlabel("Year-Month")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # 1.3. Hourly missing patterns
        if "pm2.5" in self.df.columns:
            hourly_missing = self.df.groupby(self.df.index.hour)["pm2.5"].apply(
                lambda x: x.isnull().mean()
            )
            hourly_missing.plot(kind="bar", ax=axes[1, 0])
            axes[1, 0].set_title("PM2.5 Missing Rate by Hour")
            axes[1, 0].set_xlabel("Hour of Day")

        # 1.4. Missing data correlation
        if available_cols_missing:
            missing_corr = missing_data[available_cols_missing].corr()
            sns.heatmap(
                missing_corr, annot=True, cmap="coolwarm", center=0, ax=axes[1, 1]
            )
            axes[1, 1].set_title("Missing Data Correlation")

        plt.tight_layout()
        plt.savefig(plots_path / "missing_data_patterns.pdf", bbox_inches="tight")
        plt.close(fig)

    def _plot_basic_distributions(self, plots_path: Path):
        """Plots distributions of key variables and saves to PDF."""
        print("Generating key variable distributions plot...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        key_vars_dist = ["pm2.5", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        available_vars_dist = [col for col in key_vars_dist if col in numeric_cols]

        if available_vars_dist:
            n_vars = len(available_vars_dist)
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

            axes = axes.flatten() if n_vars > 1 else [axes]

            for i, col in enumerate(available_vars_dist):
                data = self.df[col].dropna()
                axes[i].hist(data, bins=50, alpha=0.7, edgecolor="black")
                axes[i].set_title(f"{col} Distribution")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frequency")
                axes[i].axvline(
                    data.mean(),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {data.mean():.2f}",
                )
                axes[i].axvline(
                    data.median(),
                    color="green",
                    linestyle="--",
                    label=f"Median: {data.median():.2f}",
                )
                axes[i].legend()

            for i in range(n_vars, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(plots_path / "variable_distributions.pdf", bbox_inches="tight")
            plt.close(fig)

    def _plot_temporal_patterns(self, plots_path: Path):
        """Plots temporal patterns in PM2.5 and saves to PDF."""
        print("Generating temporal patterns plot...")
        if "pm2.5" in self.df.columns and self.df.index.name == "datetime":
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))

            # 1. Time series plot (sample)
            sample_data = self.df["pm2.5"].iloc[: 24 * 30]
            sample_data.plot(ax=axes[0, 0])
            axes[0, 0].set_title("PM2.5 Time Series (First Month)")
            axes[0, 0].set_ylabel("PM2.5 (μg/m³)")

            # 2. Monthly average pattern
            monthly_avg = self.df.groupby(self.df.index.month)["pm2.5"].mean()
            monthly_avg.plot(kind="bar", ax=axes[0, 1])
            axes[0, 1].set_title("Average PM2.5 by Month")
            axes[0, 1].set_xlabel("Month")

            # 3. Hourly average pattern
            hourly_avg = self.df.groupby(self.df.index.hour)["pm2.5"].mean()
            hourly_avg.plot(kind="bar", ax=axes[1, 0])
            axes[1, 0].set_title("Average PM2.5 by Hour")
            axes[1, 0].set_xlabel("Hour")

            # 4. Day of week pattern
            dow_avg = self.df.groupby(self.df.index.dayofweek)["pm2.5"].mean()
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow_avg.index = dow_names
            dow_avg.plot(kind="bar", ax=axes[1, 1])
            axes[1, 1].set_title("Average PM2.5 by Day of Week")

            # 5. Yearly trend
            yearly_avg = self.df.groupby(self.df.index.year)["pm2.5"].mean()
            yearly_avg.plot(kind="line", marker="o", ax=axes[2, 0])
            axes[2, 0].set_title("Average PM2.5 by Year")
            axes[2, 0].set_xlabel("Year")

            # 6. Cyclical feature validation
            scatter = axes[2, 1].scatter(
                self.df["hour_cos"],
                self.df["hour_sin"],
                c=self.df["hour"],
                cmap="hsv",
                alpha=0.1,
            )
            axes[2, 1].set_title("Cyclical Hour Encoding Validation")
            axes[2, 1].set_xlabel("Hour Cosine")
            axes[2, 1].set_ylabel("Hour Sine")
            plt.colorbar(scatter, ax=axes[2, 1], label="Hour")

            plt.tight_layout()
            plt.savefig(plots_path / "temporal_patterns.pdf", bbox_inches="tight")
            plt.close(fig)

    def suggest_missing_data_strategy(self):
        """Suggest missing data handling strategy"""
        print("\n" + "=" * 50)
        print("MISSING DATA STRATEGY RECOMMENDATIONS")
        print("=" * 50)

        if self.missing_summary is None:
            print("Run analyze_missing_data() first")
            return

        high_missing = self.missing_summary[
            self.missing_summary["Missing_Percentage"] > 30
        ]
        medium_missing = self.missing_summary[
            (self.missing_summary["Missing_Percentage"] > 5)
            & (self.missing_summary["Missing_Percentage"] <= 30)
        ]
        low_missing = self.missing_summary[
            (self.missing_summary["Missing_Percentage"] > 0)
            & (self.missing_summary["Missing_Percentage"] <= 5)
        ]

        if len(high_missing) > 0:
            print("HIGH MISSING (>30%):")
            for col in high_missing.index:
                print(f"  {col}: Consider dropping or external data sources")

        if len(medium_missing) > 0:
            print("\nMEDIUM MISSING (5-30%):")
            for col in medium_missing.index:
                if "pm2.5" in col.lower():
                    print(
                        f"  {col}: Forward fill + interpolation for short gaps, exclude long gaps"
                    )
                else:
                    print(
                        f"  {col}: Seasonal/trend interpolation or model-based imputation"
                    )

        if len(low_missing) > 0:
            print("\nLOW MISSING (<5%):")
            for col in low_missing.index:
                print(f"  {col}: Forward fill or linear interpolation")

        print("\nRecommended approach for PM2.5 forecasting:")
        print("1. Forward fill gaps < 3 hours")
        print("2. Linear interpolation for gaps 3-12 hours")
        print("3. Exclude training samples with gaps > 12 hours")
        print("4. Use meteorological features for longer gap imputation")

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

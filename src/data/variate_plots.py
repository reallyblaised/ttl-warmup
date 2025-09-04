"""Enhanced suite of feature- and target-variable plotting functions for time series analysis.

Overview of variables found in the Beijing PM2.5 dataset:
========================================================

Raw Variables (from the original dataset)
----------------------------------------
- No: Row number (a unique identifier, not a feature).
- year, month, day, hour: Date and time components.
- pm2.5: PM2.5 concentration in μg/m^3 (the target variable).
- DEWP: Dew Point in °C.
- TEMP: Temperature in °C.
- PRES: Atmospheric Pressure in hPa.
- cbwd: Combined Wind Direction (categorical: e.g., 'SE', 'NW', 'cv').
- Iws: Cumulated wind speed in m/s.
- Is: Cumulated hours of snow (binary flag, 1 if snow, 0 otherwise).
- Ir: Cumulated hours of rain (binary flag, 1 if rain, 0 otherwise).

Engineered Variables (for model training)
----------------------------------------
- datetime: The combined year, month, day, and hour into a single, time-series index.
- Cyclical Features:
    - hour_sin, hour_cos: Continuous representation of the hour of the day.
    - month_sin, month_cos: Continuous representation of the month of the year.
    - dow_sin, dow_cos: Continuous representation of the day of the week.
- Wind Features:
    - is_calm: A binary flag (1 or 0) indicating if the wind is calm.
    - cbwd_sin, cbwd_cos: Continuous representation of the wind direction as a vector.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scienceplots
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import warnings
from scipy import stats


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior"""

    figsize_per_var: Tuple[int, int] = (15, 4)
    target_color: str = "tab:red"
    feature_color: str = "black"  # Black
    engineered_color: str = "tab:blue"
    style_context: List[str] = field(default_factory=lambda: ["science"])
    dpi: int = 300
    rolling_window: int = 24 * 7  # 7 days for hourly data
    alpha_ci: float = 0.2  # Confidence interval alpha


class VariateVisualizer:
    """
    Enhanced class for generating comprehensive time series plots with statistical overlays.
    """

    def __init__(self, df: pd.DataFrame, config: PlotConfig = None):
        """
        Initialize the enhanced visualizer with a preprocessed pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A preprocessed DataFrame with a datetime index and engineered features.
        config : PlotConfig, optional
            Configuration object for plot appearance
        """
        self._validate_input(df)
        self.df = df
        self.config = config or PlotConfig()

        # Air quality thresholds (WHO guidelines)
        self.pm25_thresholds = {
            "Good": 12,
            "Moderate": 35.4,
            "Unhealthy for Sensitive": 55.4,
            "Unhealthy": 150.4,
            "Very Unhealthy": 250.4,
            "Hazardous": 500,
        }

    def _validate_input(self, df: pd.DataFrame):
        """Validate input DataFrame for time series analysis"""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame.")

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError(
                "DataFrame must have datetime index for time series analysis"
            )

    def _categorize_variables(
        self, variables: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Categorize variables by type for different styling"""
        if variables:
            available_vars = [col for col in variables if col in self.df.columns]
        else:
            df_numeric = self.df.select_dtypes(include=np.number)
            redundant_cols = ["year", "month", "day", "hour", "No"]
            available_vars = [
                col for col in df_numeric.columns if col not in redundant_cols
            ]

        categorized = {"target": [], "raw_features": [], "engineered_features": []}

        engineered_patterns = ["_sin", "_cos", "is_", "dow_", "hour_", "month_"]

        for var in available_vars:
            if var == "pm2.5":
                categorized["target"].append(var)
            elif any(pattern in var for pattern in engineered_patterns):
                categorized["engineered_features"].append(var)
            else:
                categorized["raw_features"].append(var)

        return categorized

    def _get_variable_color(self, var_name: str) -> str:
        """Get appropriate color for variable based on type"""
        if var_name == "pm2.5":
            return self.config.target_color
        elif any(
            pattern in var_name
            for pattern in ["_sin", "_cos", "is_", "dow_", "hour_", "month_"]
        ):
            return self.config.engineered_color
        else:
            return self.config.feature_color

    def _add_statistical_overlays(self, ax, series: pd.Series, color: str):
        """Add rolling statistics and trend overlays"""
        if len(series.dropna()) < self.config.rolling_window * 2:
            return  # Not enough data for meaningful rolling stats

        # Rolling statistics
        rolling_mean = series.rolling(
            window=self.config.rolling_window, center=True
        ).mean()
        rolling_std = series.rolling(
            window=self.config.rolling_window, center=True
        ).std()

        # Plot rolling mean
        ax.plot(
            rolling_mean.index,
            rolling_mean.values,
            color="orange",
            alpha=0.8,
            linewidth=2,
            label=f"{self.config.rolling_window//24}-day MA",
        )

        # Confidence interval
        ax.fill_between(
            rolling_mean.index,
            rolling_mean - 1.96 * rolling_std,
            rolling_mean + 1.96 * rolling_std,
            alpha=self.config.alpha_ci,
            color="orange",
            label="95% CI",
        )

    def _add_pm25_thresholds(self, ax):
        """Add WHO air quality thresholds for PM2.5"""
        colors = ["green", "yellow", "orange", "red", "purple", "maroon"]
        for i, (level, threshold) in enumerate(self.pm25_thresholds.items()):
            ax.axhline(
                y=threshold,
                color=colors[i],
                linestyle="--",
                alpha=0.6,
                linewidth=1,
                label=f"{level}: {threshold}",
            )

    def generate_variate_plots(
        self,
        variables: Optional[List[str]] = None,
        target_variable: str = "pm2.5",
        output_dir: Union[Path, str] = "plots/eda",
        add_overlays: bool = True,
        save_format: str = "pdf",
    ) -> None:
        """
        Generate enhanced time series plots with statistical overlays.

        Parameters
        ----------
        variables : Optional[List[str]]
            Variables to plot. If None, plots all numeric columns.
        target_variable : str
            Name of target variable for special highlighting
        output_dir : Union[Path, str]
            Directory to save plots
        add_overlays : bool
            Whether to add statistical overlays
        save_format : str
            File format for saving ('pdf', 'png', 'svg')
        """
        categorized = self._categorize_variables(variables)
        all_vars = (
            categorized["target"]
            + categorized["raw_features"]
            + categorized["engineered_features"]
        )

        if not all_vars:
            warnings.warn("No variables found to plot.")
            return

        plots_path = Path(output_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        n_vars = len(all_vars)
        fig, axes = plt.subplots(n_vars, 1, figsize=(15, 4 * n_vars), sharex=True)

        if n_vars == 1:
            axes = [axes]

        with plt.style.context(self.config.style_context):
            for i, col in enumerate(all_vars):
                color = self._get_variable_color(col)

                # Main time series plot
                self.df[col].plot(ax=axes[i], color=color, alpha=0.7, linewidth=1)

                # Add statistical overlays
                if add_overlays:
                    self._add_statistical_overlays(axes[i], self.df[col], color)

                # Add PM2.5 thresholds if target variable
                if col == target_variable and add_overlays:
                    self._add_pm25_thresholds(axes[i])

                axes[i].set_title(f"Time Series: {col}", fontsize=12, fontweight="bold")
                axes[i].set_ylabel(col, fontsize=10)
                axes[i].grid(True, alpha=0.3)
                if add_overlays:
                    axes[i].legend(loc="upper right", fontsize=8)

            axes[-1].set_xlabel("Time", fontsize=10)
            fig.suptitle(
                "Enhanced Variable Time Series Analysis",
                y=1.02,
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save plot
            filename = f"enhanced_variate_plots.{save_format}"
            plt.savefig(plots_path / filename, bbox_inches="tight", dpi=self.config.dpi)
            plt.close(fig)

        print(f"Enhanced variate plots saved to {plots_path.resolve()}")

    def plot_correlation_matrix(
        self,
        variables: Optional[List[str]] = None,
        output_dir: Union[Path, str] = "plots/eda",
        method: str = "pearson",
    ) -> None:
        """Generate correlation heatmap for variable relationships"""
        categorized = self._categorize_variables(variables)
        all_vars = (
            categorized["target"]
            + categorized["raw_features"]
            + categorized["engineered_features"]
        )

        if len(all_vars) < 2:
            warnings.warn("Need at least 2 variables for correlation analysis.")
            return

        plots_path = Path(output_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        # Calculate correlation matrix
        corr_matrix = self.df[all_vars].corr(method=method)

        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        with plt.style.context(self.config.style_context):
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap="RdBu_r",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                fmt=".2f",
            )

            plt.title(
                f"Variable Correlation Matrix ({method.title()})",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            plt.savefig(
                plots_path / f"correlation_matrix_{method}.pdf",
                bbox_inches="tight",
                dpi=self.config.dpi,
            )
            plt.close()

        print(f"Correlation matrix saved to {plots_path.resolve()}")

    def plot_seasonal_decomposition(
        self,
        target_var: str = "pm2.5",
        output_dir: Union[Path, str] = "plots/eda",
        period: Optional[int] = None,
    ) -> None:
        """Plot seasonal decomposition of target variable"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            raise ImportError("statsmodels required for seasonal decomposition")

        if target_var not in self.df.columns:
            raise ValueError(f"Variable {target_var} not found in DataFrame")

        plots_path = Path(output_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        # Infer period if not provided (assume daily seasonality for hourly data)
        if period is None:
            freq = pd.infer_freq(self.df.index)
            period = 24 if freq and "H" in freq else 7

        # Remove missing values for decomposition
        series = self.df[target_var].dropna()

        if len(series) < 2 * period:
            warnings.warn(
                f"Not enough data for seasonal decomposition with period {period}"
            )
            return

        # Perform decomposition
        decomposition = seasonal_decompose(series, model="additive", period=period)

        # Create plot
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        with plt.style.context(self.config.style_context):
            # Original series
            decomposition.observed.plot(ax=axes[0], color=self.config.target_color)
            axes[0].set_title(f"Original {target_var}", fontweight="bold")
            axes[0].grid(True, alpha=0.3)

            # Trend
            decomposition.trend.plot(ax=axes[1], color="orange")
            axes[1].set_title("Trend", fontweight="bold")
            axes[1].grid(True, alpha=0.3)

            # Seasonal
            decomposition.seasonal.plot(ax=axes[2], color="green")
            axes[2].set_title("Seasonal", fontweight="bold")
            axes[2].grid(True, alpha=0.3)

            # Residual
            decomposition.resid.plot(ax=axes[3], color="red")
            axes[3].set_title("Residual", fontweight="bold")
            axes[3].grid(True, alpha=0.3)

            plt.suptitle(
                f"Seasonal Decomposition: {target_var}", fontsize=16, fontweight="bold"
            )
            plt.tight_layout()

            plt.savefig(
                plots_path / f"seasonal_decomposition_{target_var}.pdf",
                bbox_inches="tight",
                dpi=self.config.dpi,
            )
            plt.close()

        print(f"Seasonal decomposition saved to {plots_path.resolve()}")

    def plot_data_quality_report(
        self, output_dir: Union[Path, str] = "plots/eda"
    ) -> None:
        """Generate comprehensive data quality report"""
        plots_path = Path(output_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        with plt.style.context(self.config.style_context):
            # Missing data heatmap
            missing_data = self.df[numeric_cols].isnull()
            if missing_data.any().any():
                sns.heatmap(missing_data.T, cmap="viridis", cbar=True, ax=axes[0, 0])
                axes[0, 0].set_title("Missing Data Pattern", fontweight="bold")
            else:
                axes[0, 0].text(
                    0.5, 0.5, "No Missing Data", ha="center", va="center", fontsize=14
                )
                axes[0, 0].set_title("Missing Data Pattern", fontweight="bold")

            # Distribution of missing values
            missing_counts = self.df[numeric_cols].isnull().sum()
            missing_counts[missing_counts > 0].plot(kind="bar", ax=axes[0, 1])
            axes[0, 1].set_title("Missing Value Counts", fontweight="bold")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # Outlier detection (IQR method)
            outlier_counts = []
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (self.df[col] < (Q1 - 1.5 * IQR))
                    | (self.df[col] > (Q3 + 1.5 * IQR))
                ).sum()
                outlier_counts.append(outliers)

            outlier_df = pd.DataFrame(
                {"Variable": numeric_cols, "Outliers": outlier_counts}
            )
            outlier_df = outlier_df[outlier_df["Outliers"] > 0].sort_values("Outliers")

            if not outlier_df.empty:
                outlier_df.plot(x="Variable", y="Outliers", kind="bar", ax=axes[1, 0])
                axes[1, 0].set_title("Outlier Counts (IQR Method)", fontweight="bold")
                axes[1, 0].tick_params(axis="x", rotation=45)
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No Outliers Detected",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                axes[1, 0].set_title("Outlier Counts (IQR Method)", fontweight="bold")

            # Data completeness over time
            completeness = (
                self.df[numeric_cols]
                .notna()
                .mean(axis=1)
                .rolling(window=self.config.rolling_window)
                .mean()
            )
            completeness.plot(ax=axes[1, 1], color="blue")
            axes[1, 1].set_title("Data Completeness Over Time", fontweight="bold")
            axes[1, 1].set_ylabel("Completeness Ratio")
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle("Data Quality Report", fontsize=16, fontweight="bold")
            plt.tight_layout()

            plt.savefig(
                plots_path / "data_quality_report.pdf",
                bbox_inches="tight",
                dpi=self.config.dpi,
            )
            plt.close()

        print(f"Data quality report saved to {plots_path.resolve()}")

    def validate_cyclical_features(
        self, output_dir: Union[Path, str] = "plots/eda"
    ) -> None:
        """Validate cyclical feature engineering"""
        plots_path = Path(output_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        cyclical_pairs = [
            ("hour_sin", "hour_cos"),
            ("month_sin", "month_cos"),
            ("dow_sin", "dow_cos"),
        ]

        available_pairs = [
            (sin_col, cos_col)
            for sin_col, cos_col in cyclical_pairs
            if sin_col in self.df.columns and cos_col in self.df.columns
        ]

        if not available_pairs:
            warnings.warn("No cyclical feature pairs found for validation")
            return

        fig, axes = plt.subplots(
            len(available_pairs), 2, figsize=(12, 4 * len(available_pairs))
        )

        if len(available_pairs) == 1:
            axes = axes.reshape(1, -1)

        with plt.style.context(self.config.style_context):
            for i, (sin_col, cos_col) in enumerate(available_pairs):
                # Check unit circle property: sin²+cos²=1
                magnitude = np.sqrt(self.df[sin_col] ** 2 + self.df[cos_col] ** 2)

                # Unit circle scatter plot
                axes[i, 0].scatter(
                    self.df[sin_col],
                    self.df[cos_col],
                    alpha=0.6,
                    s=1,
                    color=self.config.engineered_color,
                )
                circle = plt.Circle((0, 0), 1, fill=False, color="red", linestyle="--")
                axes[i, 0].add_patch(circle)
                axes[i, 0].set_xlim(-1.2, 1.2)
                axes[i, 0].set_ylim(-1.2, 1.2)
                axes[i, 0].set_aspect("equal")
                axes[i, 0].set_title(f"{sin_col} vs {cos_col}", fontweight="bold")
                axes[i, 0].grid(True, alpha=0.3)

                # Magnitude validation
                axes[i, 1].hist(
                    magnitude, bins=50, alpha=0.7, color=self.config.engineered_color
                )
                axes[i, 1].axvline(
                    x=1.0, color="red", linestyle="--", label="Expected: 1.0"
                )
                axes[i, 1].set_title(
                    f"Magnitude Distribution\nMean: {magnitude.mean():.4f}",
                    fontweight="bold",
                )
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)

            plt.suptitle("Cyclical Feature Validation", fontsize=16, fontweight="bold")
            plt.tight_layout()

            plt.savefig(
                plots_path / "cyclical_feature_validation.pdf",
                bbox_inches="tight",
                dpi=self.config.dpi,
            )
            plt.close()

        print(f"Cyclical feature validation saved to {plots_path.resolve()}")

    def generate_comprehensive_report(
        self,
        variables: Optional[List[str]] = None,
        target_variable: str = "pm2.5",
        output_dir: Union[Path, str] = "plots/eda",
    ) -> None:
        """Generate all visualization reports"""
        print("Generating comprehensive EDA report...")

        # Core time series plots
        self.generate_variate_plots(variables, target_variable, output_dir)

        # Statistical analysis
        self.plot_correlation_matrix(variables, output_dir)

        # Time series specific analysis
        if target_variable in self.df.columns:
            self.plot_seasonal_decomposition(target_variable, output_dir)

        # Data quality assessment
        self.plot_data_quality_report(output_dir)

        # Feature engineering validation
        self.validate_cyclical_features(output_dir)

        print(
            f"Comprehensive EDA report completed. All plots saved to {Path(output_dir).resolve()}"
        )


# # Example usage and configuration
# if __name__ == "__main__":
#     # Example configuration
#     config = PlotConfig(
#         target_color="tab:blue",
#         feature_color="black",  # Black for non-target variables
#         engineered_color="tab:blue",
#         rolling_window=24 * 7,  # Weekly rolling window for hourly data
#         dpi=300,
#     )

#     Initialize visualizer (assuming df is your Beijing PM2.5 DataFrame)
#     visualizer = VariateVisualizer(df, config)
#     visualizer.generate_comprehensive_report()

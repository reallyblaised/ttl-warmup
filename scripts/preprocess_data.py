"""
Perform the data exploration and preprocessing for the Beijing PM2.5 dataset, inheriting from the BeijingPM25Explorer class.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # HACK

from src.data.preprocessor import BeijingPM25Explorer
from src.data.variate_plots import VariateVisualizer


def main():
    """Run data exploration and preprocessing"""
    data_path = "data/raw/beijing_pm25_raw.csv"

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Run ./scripts/download_data.sh first."
        )

    print("Starting Beijing PM2.5 data exploration...")

    # Initialize explorer
    explorer = BeijingPM25Explorer(data_path)

    # Load and inspect
    df = explorer.load_and_inspect()

    # datetime index and cyclic encoding
    df = explorer.feature_transform()

    # Analyze missing data
    missing_summary = explorer.analyze_missing_data()

    # Get data summary
    data_summary = explorer.get_data_summary()
    print(f"\nData Summary: {data_summary}")

    # Engage EDA by way of visualizing the variates
    visualizer = VariateVisualizer(df)
    visualizer.generate_comprehensive_report()

    # Save processed data for next steps
    processed_path = "data/processed"

    Path(processed_path).mkdir(parents=True, exist_ok=True)
    df.to_pickle(f"{processed_path}/beijing_pm25.pkl")
    print(f"\nProcessed data saved to: {processed_path}/beijing_pm25.pkl")

    return 0


if __name__ == "__main__":
    exit(main())

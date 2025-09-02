"""
Perform the data exploration and preprocessing for the Beijing PM2.5 dataset, inheriting from the BeijingPM25Explorer class.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # HACK

from src.data.preprocessor import BeijingPM25Explorer


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

    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        explorer.plot_missing_patterns()
        explorer.plot_basic_distributions()
        explorer.plot_temporal_patterns()
    except Exception as e:
        print(f"Warning: Plotting failed: {e}")
        print("Continuing with analysis...")

    # # Get recommendations
    # explorer.suggest_missing_data_strategy()

    # # Save processed data for next steps
    # processed_path = "data/processed/beijing_pm25_explored.pkl"
    # os.makedirs("data/processed", exist_ok=True)
    # df.to_pickle(processed_path)
    # print(f"\nProcessed data saved to: {processed_path}")

    return 0


if __name__ == "__main__":
    exit(main())

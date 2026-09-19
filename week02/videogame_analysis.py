from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# Configuration

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE_CANDIDATES = [
    PROJECT_DIR / "vgsales.csv",
    PROJECT_DIR / "data" / "vgsales.csv",
]
OUTPUT_DIR = PROJECT_DIR / "output"

REQUIRED_COLUMNS = {
    "Name",
    "Platform",
    "Genre",
    "Global_Sales",
}

# Data loading and preparation

def find_data_file() -> Path:
    """This will return the vgsales.csv file"""
    for file_path in DATA_FILE_CANDIDATES:
        if file_path.exists():
            return file_path

    locations = "\n".join(f"  - {path}" for path in DATA_FILE_CANDIDATES)
    raise FileNotFoundError("Could not find the referenced file")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Convert the CSV file into a DataFrame for better analysis"""
    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file is empty") from error
    except pd.errors.ParserError as error:
        raise ValueError("The CSV file was not parsed") from error

    return data


def validate_columns(data: pd.DataFrame) -> None:
    """Validate that the csv file has the requried columns or check if therer is one missing"""
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            "There is a missing colum - " f"{missing_text}"
        )


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Remove the fields that are empty and provides the dataframe"""
    cleaned = data.copy()

    # Convert Global_Sales to numbers
    cleaned["Global_Sales"] = pd.to_numeric(
        cleaned["Global_Sales"], errors="coerce"
    )

    # Remove records that cannot be used
    cleaned = cleaned.dropna(
        subset=["Genre", "Platform", "Global_Sales"]
    )

    # Remove extra spaces
    cleaned["Genre"] = cleaned["Genre"].astype(str).str.strip()
    cleaned["Platform"] = cleaned["Platform"].astype(str).str.strip()

    # Only keep rows with valid names
    cleaned = cleaned[
        (cleaned["Genre"] != "")
        & (cleaned["Platform"] != "")
        & (cleaned["Global_Sales"] >= 0)
    ]

    return cleaned


def print_dataset_summary(original: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    """Display a summary of dataframe"""
    print("\nDATASET SUMMARY")
    print("-" * 60)
    print(f"Original rows: {len(original):,}")
    print(f"Rows used in analysis: {len(cleaned):,}")
    print(f"Unique genres: {cleaned['Genre'].nunique()}")
    print(f"Unique platforms: {cleaned['Platform'].nunique()}")
    print(
        "Total global sales represented: "
        f"{cleaned['Global_Sales'].sum():,.2f} million units"
    )



# Question 1 - Genre sales

def analyze_genre_sales(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate total global sales and game counts for every genre"""
    genre_results = (
        data.groupby("Genre", as_index=False)
        .agg(
            Total_Global_Sales=("Global_Sales", "sum"),
            Game_Count=("Name", "count"),
        )
        .sort_values("Total_Global_Sales", ascending=False)
        .reset_index(drop=True)
    )

    genre_results["Average_Sales_Per_Game"] = (
        genre_results["Total_Global_Sales"] / genre_results["Game_Count"]
    )

    return genre_results


def print_genre_results(results: pd.DataFrame, top_n: int = 10) -> None:
    """Display the genres that has the better global sales"""
    print("\nQUESTION 1")
    print("Which genres have the highest total global sales?")
    print("-" * 60)

    display_columns = ["Genre", "Total_Global_Sales", "Game_Count"]
    print(
        results[display_columns]
        .head(top_n)
        .to_string(index=False, formatters={
            "Total_Global_Sales": "{:,.2f}".format,
        })
    )

    top_genre = results.iloc[0]
    print(
        f"\nAnswer: {top_genre['Genre']} is the highest-selling genre " f"with {top_genre['Total_Global_Sales']:,.2f} million units in " "combined global sales."
    )

# Question 2 - Platform average sales

def analyze_platform_average_sales(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate total, average, and game count for each platform that is available"""
    platform_results = (
        data.groupby("Platform", as_index=False)
        .agg(
            Average_Global_Sales=("Global_Sales", "mean"),
            Total_Global_Sales=("Global_Sales", "sum"),
            Game_Count=("Name", "count"),
        )
        .sort_values("Average_Global_Sales", ascending=False)
        .reset_index(drop=True)
    )

    return platform_results


def print_platform_results(results: pd.DataFrame, top_n: int = 10) -> None:
    """ Display the platforms by the average of global sales per game."""
    print("\nQUESTION 2")
    print("Which platforms have the highest average global sales per game?")
    print("-" * 60)

    print(
        results.head(top_n).to_string(
            index=False,
            formatters={
                "Average_Global_Sales": "{:,.3f}".format,
                "Total_Global_Sales": "{:,.2f}".format,
            },
        )
    )

    top_platform = results.iloc[0]
    print(
        f"\nAnswer: {top_platform['Platform']} has the highest average " f"global sales at {top_platform['Average_Global_Sales']:,.2f} " "million units per game in this dataset."
    )


# Graphs

def create_genre_chart(results: pd.DataFrame, output_dir: Path) -> Path:
    """Create and save a bar chart of total global sales by genre."""
    chart_data = results.sort_values("Total_Global_Sales", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(chart_data["Genre"], chart_data["Total_Global_Sales"])
    ax.set_title("Total Global Video Game Sales by Genre")
    ax.set_xlabel("Global Sales (millions of units)")
    ax.set_ylabel("Genre")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / "genre_global_sales.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def create_platform_chart(
    results: pd.DataFrame, output_dir: Path, top_n: int = 10) -> Path:
    """Create chart of the top platforms by average sales per game"""
    chart_data = (
        results.head(top_n)
        .sort_values("Average_Global_Sales", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(chart_data["Platform"], chart_data["Average_Global_Sales"])
    ax.set_title(f"Top {top_n} Platforms by Average Global Sales per Game")
    ax.set_xlabel("Average Global Sales (millions of units per game)")
    ax.set_ylabel("Platform")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / "platform_average_sales.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path

# Output files

def save_result_tables(
    genre_results: pd.DataFrame,
    platform_results: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save files as CSV files."""
    genre_path = output_dir / "genre_sales_results.csv"
    platform_path = output_dir / "platform_average_sales_results.csv"

    genre_results.to_csv(genre_path, index=False)
    platform_results.to_csv(platform_path, index=False)

    return genre_path, platform_path


def save_text_report(
    genre_results: pd.DataFrame,
    platform_results: pd.DataFrame,
    output_dir: Path,) -> Path:
    """Save a concise text report containing the answers to both questions."""
    top_genre = genre_results.iloc[0]
    top_platform = platform_results.iloc[0]

    report = [
        "VIDEO GAME SALES DATA ANALYSIS",
        "=" * 60,
        "",
        "Question 1: Which genres have the highest total global sales?", f"Top genre: {top_genre['Genre']}",
        (
            "Total global sales: " f"{top_genre['Total_Global_Sales']:,.2f} million units"),
        "",
        "Top 5 genres:",
        genre_results[["Genre", "Total_Global_Sales"]].head(5).to_string(index=False),
        "",
        "Question 2: Which platforms have the highest average global sales per game?", f"Top platform: {top_platform['Platform']}",
        (
            "Average global sales per game: " f"{top_platform['Average_Global_Sales']:,.3f} million units"
        ),
        "",
        "Top 5 platforms:",
        platform_results[
            ["Platform", "Average_Global_Sales", "Game_Count"]].head(5).to_string(index=False),
        "",
    ]

    output_path = output_dir / "analysis_report.txt"
    output_path.write_text("\n".join(report), encoding="utf-8")
    return output_path

# Program main function

def main() -> int:
    """Run analysis workflow."""
    print("Video Game Sales Data Analysis")
    print("=" * 60)

    try:
        data_file = find_data_file()
        print(f"Using dataset: {data_file}")

        original_data = load_dataset(data_file)
        validate_columns(original_data)
        clean_data = clean_dataset(original_data)

        if clean_data.empty:
            raise ValueError("No usable records remained after data cleaning.")

        print_dataset_summary(original_data, clean_data)

        genre_results = analyze_genre_sales(clean_data)
        platform_results = analyze_platform_average_sales(clean_data)

        print_genre_results(genre_results)
        print_platform_results(platform_results)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        genre_table, platform_table = save_result_tables(
            genre_results, platform_results, OUTPUT_DIR
        )
        genre_chart = create_genre_chart(genre_results, OUTPUT_DIR)
        platform_chart = create_platform_chart(platform_results, OUTPUT_DIR)
        report_path = save_text_report(
            genre_results, platform_results, OUTPUT_DIR
        )

        print("\nFILES CREATED")
        print("-" * 60)
        print(f"Genre results: {genre_table}")
        print(f"Platform results: {platform_table}")
        print(f"Genre chart: {genre_chart}")
        print(f"Platform chart: {platform_chart}")
        print(f"Text report: {report_path}")
        print("\nAnalysis completed successfully.")
        return 0

    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        return 1

    except Exception as error: 
        print(f"\nUnexpected error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

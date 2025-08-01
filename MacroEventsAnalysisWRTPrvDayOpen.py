'''

- TO FIND OUT WHAT IF BUYING/GOING LONG ON PRV DAY OPEN WRT EVENT DAY, YIELDS ANY RETURNS ON AVG;

- RETURNS FROM EVENT DAY CLOSE AND PRV DAY OPEN RELATED PIVOT LEVEL % RETURNS


'''


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import re

month_patterns = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' \
                 r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b'

warnings.filterwarnings('ignore')


def load_economic_events(file_path):
    """
    Load economic events data from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        # Clean column names
        df.columns = df.columns.str.strip()

        print("Available columns in the file:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())

        # Convert datetime column - try different possible column names
        datetime_cols = ['Datetime', 'datetime', 'Date', 'date']
        datetime_col = None
        for col in datetime_cols:
            if col in df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            print("No datetime column found. Available columns:", df.columns.tolist())
            return None

        df['Datetime'] = pd.to_datetime(df[datetime_col])

        # Extract unique events that occur monthly - try different possible column names
        event_cols = ['importance event', 'event', 'Event', 'importance_event', 'event_name']
        event_col = None
        for col in event_cols:
            if col in df.columns:
                event_col = col
                break

        if event_col is None:
            print("No event column found. Available columns:", df.columns.tolist())
            return None

        df['event_name'] = df[event_col].astype(str).str.strip()

        df['event_name'] = df['event_name'].apply(lambda x: re.sub(month_patterns, '', x, flags=re.IGNORECASE).strip())

        # Remove leftover empty parentheses and dashes
        df['event_name'] = df['event_name'].str.replace(r'\(\s*\)', '', regex=True)
        df['event_name'] = df['event_name'].str.replace(r'[-–]\s*$', '', regex=True)

        # Final cleanup: remove extra whitespace
        df['event_name'] = df['event_name'].str.strip()

        # Filter out any null or empty event names
        df = df[df['event_name'].notna() & (df['event_name'] != '') & (df['event_name'] != 'nan')]

        print(f"\nUnique events found: {df['event_name'].nunique()}")
        print("Sample events:")
        print(df['event_name'].value_counts().head(10))

        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Please check the file path and make sure the file exists.")
        return None


def download_nasdaq_data(start_date, end_date):
    """
    Download NASDAQ data from yfinance
    """
    try:
        nasdaq = yf.download('^IXIC', start=start_date, end=end_date, progress=False)

        if isinstance(nasdaq.columns, pd.MultiIndex):
            # Flatten MultiIndex: ('Open', '^IXIC') → 'Open'
            nasdaq.columns = [col[0] for col in nasdaq.columns]

        nasdaq.reset_index(inplace=True)
        return nasdaq
    except Exception as e:
        print(f"Error downloading NASDAQ data: {e}")
        return None


def calculate_pivot_level(nasdaq_data, event_date):
    """
    Calculate pivot level: avg of (close of day i-2) and (open of day i-1)
    where event_date is day i
    """
    try:
        event_date = pd.to_datetime(event_date).date()

        # Find the event date in nasdaq data
        nasdaq_data['Date'] = pd.to_datetime(nasdaq_data['Date']).dt.date

        # Get dates for calculation
        available_dates = sorted(nasdaq_data['Date'].unique())

        if event_date not in available_dates:
            # Find the closest trading day
            closest_date = min(available_dates, key=lambda x: abs((x - event_date).days))
            event_date = closest_date

        event_idx = available_dates.index(event_date)

        if event_idx < 2:
            return None, None, None

        # Get day i-2 (2 trading days before)
        day_i_minus_2 = available_dates[event_idx - 2]
        close_i_minus_2 = nasdaq_data[nasdaq_data['Date'] == day_i_minus_2]['Close'].iloc[0]

        # Get day i-1 (1 trading day before)
        day_i_minus_1 = available_dates[event_idx - 1]
        open_i_minus_1 = nasdaq_data[nasdaq_data['Date'] == day_i_minus_1]['Open'].iloc[0]

        # Calculate pivot level
        pivot_level = (close_i_minus_2 + open_i_minus_1) / 2

        # Get event day data
        event_day_data = nasdaq_data[nasdaq_data['Date'] == event_date].iloc[0]

        return pivot_level, event_day_data, event_date

    except Exception as e:
        print(f"Error calculating pivot for {event_date}: {e}")
        return None, None, None


def analyze_event_impact(events_df, nasdaq_data):
    """
    Analyze the impact of each unique event on NASDAQ movements
    """
    results = {}

    # Get unique events
    unique_events = events_df['event_name'].unique()

    for event_name in unique_events:
        print(f"Analyzing event: {event_name}")

        event_occurrences = events_df[events_df['event_name'] == event_name]

        green_above_pivot_count = 0
        percent_distances = []

        valid_occurrences = 0

        for idx, row in event_occurrences.iterrows():
            event_date = row['Datetime']

            pivot_level, event_day_data, actual_event_date = calculate_pivot_level(nasdaq_data, event_date)

            if pivot_level is None or event_day_data is None:
                continue

            valid_occurrences += 1

            close_price = event_day_data['Close']

            # Check if close is above pivot level
            if close_price > pivot_level:
                green_above_pivot_count += 1

            # Calculate percentage distance from pivot to close
            pct_distance = ((close_price - pivot_level) / pivot_level) * 100
            percent_distances.append(pct_distance)

        if valid_occurrences > 0:
            green_above_pct = round((green_above_pivot_count / valid_occurrences) * 100, 2)
            avg_pct_distance = round(np.mean(percent_distances), 2)

            results[event_name] = {
                'total_occurrences': valid_occurrences,
                'green_above_pivot_pct': green_above_pct,
                'avg_close_distance_pct': avg_pct_distance
            }

    return results


def create_heatmap(results, output_csv="nasdaq_event_analysis.csv"):
    """
    Save simplified analysis results to CSV
    """
    if not results:
        print("No results to save")
        return None

    events = list(results.keys())
    metrics = ['Green Close Above Pivot %', 'Avg Close Distance %', 'Total Occurrences']

    heatmap_data = []
    for event in events:
        row = [
            results[event]['green_above_pivot_pct'],
            results[event]['avg_close_distance_pct'],
            results[event]['total_occurrences']
        ]
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=events, columns=metrics)
    heatmap_df.to_csv(output_csv)
    print(f"Saved heatmap data to: {output_csv}")

    return heatmap_df


def print_detailed_results(results):
    """
    Print simplified results showing percentage of green closes above pivot and avg % distance
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS RESULTS")
    print("=" * 80)

    for event_name, data in results.items():
        print(f"\nEvent: {event_name}")
        print(f"Total Valid Occurrences: {data['total_occurrences']}")
        print(f"Green Close Above Pivot: {data['green_above_pivot_pct']}%")
        print(f"Average % Distance from Pivot to Close: {data['avg_close_distance_pct']}%")
        print("-" * 50)


def debug_first_event(events_df, nasdaq_data):
    """
    Debug function to test the first event and identify issues
    """
    print("=== DEBUGGING FIRST EVENT ===")

    # Get the first event occurrence
    first_event = events_df.iloc[0]
    print(f"First event: {first_event['event_name']}")
    print(f"Event date: {first_event['Datetime']}")

    # Test pivot calculation
    pivot_level, event_day_data, actual_event_date = calculate_pivot_level(nasdaq_data, first_event['Datetime'])

    if pivot_level is None:
        print("Pivot calculation returned None")
        return

    print(f"Pivot level: {pivot_level}")
    print(f"Event day data type: {type(event_day_data)}")
    print(f"Event day data: {event_day_data}")

    if hasattr(event_day_data, 'Open'):
        print(f"Open: {event_day_data['Open']} (type: {type(event_day_data['Open'])})")
        print(f"Close: {event_day_data['Close']} (type: {type(event_day_data['Close'])})")
        print(f"Low: {event_day_data['Low']} (type: {type(event_day_data['Low'])})")

    print("=== END DEBUG ===\n")


def main():
    """
    Main function to run the analysis
    """
    # File path - update this to your file location
    file_path = "cleaned_economic_events.csv"  # Update this path

    print("Loading economic events data...")
    events_df = load_economic_events(file_path)

    if events_df is None:
        print("Failed to load events data. Please check the file path and format.")
        return None, None

    print(f"Loaded {len(events_df)} economic events")
    print(f"Date range: {events_df['Datetime'].min()} to {events_df['Datetime'].max()}")

    # Download NASDAQ data
    start_date = events_df['Datetime'].min() - timedelta(days=10)  # Extra buffer
    end_date = events_df['Datetime'].max() + timedelta(days=10)

    print(f"\nDownloading NASDAQ data from {start_date} to {end_date}...")
    nasdaq_data = download_nasdaq_data(start_date, end_date)

    if nasdaq_data is None:
        print("Failed to download NASDAQ data.")
        return None, None

    print(f"Downloaded {len(nasdaq_data)} trading days of NASDAQ data")

    # Debug first event
    debug_first_event(events_df, nasdaq_data)

    # Analyze events
    print("\nAnalyzing event impacts...")
    results = analyze_event_impact(events_df, nasdaq_data)

    if not results:
        print("No valid results found.")
        return None, None

    # Print detailed results
    print_detailed_results(results)

    # Create heatmap
    print("\nCreating heatmap visualization...")
    heatmap_df = create_heatmap(results)

    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"Total unique events analyzed: {len(results)}")

    return results, heatmap_df


# Run the analysis
if __name__ == "__main__":
    try:
        results, heatmap_data = main()
        if results is not None:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed. Please check the error messages above.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")










import pandas as pd
import yfinance as yf
import re
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

month_patterns = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' \
                 r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b'

def load_economic_events(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        datetime_cols = ['Datetime', 'datetime', 'Date', 'date']
        event_cols = ['importance event', 'event', 'Event', 'importance_event', 'event_name']

        datetime_col = next((col for col in datetime_cols if col in df.columns), None)
        event_col = next((col for col in event_cols if col in df.columns), None)

        if not datetime_col or not event_col:
            print("Missing required columns.")
            return None

        df['Datetime'] = pd.to_datetime(df[datetime_col])
        df['event_name'] = df[event_col].astype(str).str.strip()
        df['event_name'] = df['event_name'].apply(lambda x: re.sub(month_patterns, '', x, flags=re.IGNORECASE).strip())
        df['event_name'] = df['event_name'].str.replace(r'\(\s*\)', '', regex=True)
        df['event_name'] = df['event_name'].str.replace(r'[-\u2013]\s*$', '', regex=True).str.strip()
        df = df[df['event_name'].notna() & (df['event_name'] != '') & (df['event_name'] != 'nan')]

        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def download_nasdaq_data(start_date, end_date):
    try:
        nasdaq = yf.download('^IXIC', start=start_date, end=end_date, progress=False)
        if isinstance(nasdaq.columns, pd.MultiIndex):
            nasdaq.columns = [col[0] for col in nasdaq.columns]
        nasdaq.reset_index(inplace=True)
        return nasdaq
    except Exception as e:
        print(f"Error downloading NASDAQ data: {e}")
        return None

def intraday_range_by_event(events_df, nasdaq_data):
    nasdaq_data['Date'] = pd.to_datetime(nasdaq_data['Date']).dt.date
    nasdaq_data = nasdaq_data.sort_values('Date').reset_index(drop=True)

    results = []

    for event in events_df['event_name'].unique():
        df_event = events_df[events_df['event_name'] == event]
        total_days = 0
        under_2 = 0
        under_1 = 0

        for _, row in df_event.iterrows():
            event_date = pd.to_datetime(row['Datetime']).date()

            if event_date not in nasdaq_data['Date'].values:
                continue

            idx = nasdaq_data[nasdaq_data['Date'] == event_date].index[0]
            if idx < 1:
                continue

            prev_day_date = nasdaq_data.loc[idx - 1, 'Date']

            # Skip if the previous day is also an event day
            if prev_day_date in pd.to_datetime(events_df['Datetime']).dt.date.values:
                continue

            prev_day_row = nasdaq_data.loc[idx - 1]

            high = prev_day_row['High']
            low = prev_day_row['Low']

            if low == 0:
                continue

            range_pct = (high - low) / low * 100
            total_days += 1

            if range_pct < 2:
                under_2 += 1
            if range_pct < 1:
                under_1 += 1

        pct_2 = round((under_2 / total_days) * 100, 2) if total_days > 0 else 0
        pct_1 = round((under_1 / total_days) * 100, 2) if total_days > 0 else 0

        results.append({
            'event_name': event,
            'total_days': total_days,
            'under_2_pct': pct_2,
            'under_1_pct': pct_1
        })

    return pd.DataFrame(results).sort_values('total_days', ascending=False).reset_index(drop=True)

def main():
    file_path = "cleaned_economic_events.csv"  # Update this path as needed
    events_df = load_economic_events(file_path)
    if events_df is None:
        return

    start_date = events_df['Datetime'].min() - timedelta(days=10)
    end_date = events_df['Datetime'].max() + timedelta(days=10)

    nasdaq_data = download_nasdaq_data(start_date, end_date)
    if nasdaq_data is None:
        return

    intraday_event_stats_df = intraday_range_by_event(events_df, nasdaq_data)
    print("\nevent day prv day high low range so as to use delta neutral strategies :> SEE below\n")
    print("\nNote - those prv day are excluded from calculation where prv day of event is already a event itself\n")
    print(intraday_event_stats_df)

if __name__ == "__main__":
    main()




import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import re
import os
from PIL import Image, ImageDraw, ImageFont
import io

warnings.filterwarnings('ignore')


# Define helper functions first
def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


# Function to remove Chinese characters using Unicode range
def remove_chinese(text):
    if isinstance(text, str):
        return re.sub(r'[\u4e00-\u9fff]+', '', text)
    return text


# Step 1: Load event data (already in IST, timezone-naive)
df = pd.read_csv("C:/Users/DELL/Desktop/economic data/processed_economic_events_with_datetime.csv")

# Remove chinese characters etc from each cell except data
# Apply the function to all elements in the DataFrame
df = df.applymap(remove_chinese)
# if you want to see a saved copy of cleaned data
# df.to_csv("C:/Users/DELL/Desktop/economic data/cleaned_economic_events.csv", index=False)


# Remove final (Month) like (Dec), (Jan), etc. — but keep (MoM), (YoY), etc.
df['event'] = df['event'].str.replace(r'\s*\(([A-Za-z]{3})\)$', '', regex=True).str.strip()

# Convert 'Datetime' to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df = df.dropna(subset=['Datetime'])

# Step 2: Download NASDAQ data
print("Downloading NASDAQ data...")
nasdaq = yf.download('^IXIC', start='1999-12-01', end='2025-01-01', auto_adjust=False)

# Step 3: Flatten MultiIndex if needed
if isinstance(nasdaq.columns, pd.MultiIndex):
    nasdaq.columns = nasdaq.columns.get_level_values(0)

# Step 4: Convert NASDAQ datetime index from UTC to IST
nasdaq.index = nasdaq.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

# Step 5: Calculate comprehensive return metrics
print("Calculating comprehensive metrics...")

# Forward returns
for days in [1, 3, 5, 10, 30]:
    nasdaq[f'Return_{days}d'] = nasdaq['Close'].pct_change(days).shift(-days)

# Backward returns (pre-event)
for days in [1, 3, 5, 10, 30]:
    nasdaq[f'Pre_Return_{days}d'] = nasdaq['Close'].pct_change(days)

# Volatility metrics - Fixed ATR calculation
nasdaq['High_Low'] = nasdaq['High'] - nasdaq['Low']
nasdaq['High_Close'] = abs(nasdaq['High'] - nasdaq['Close'].shift(1))
nasdaq['Low_Close'] = abs(nasdaq['Low'] - nasdaq['Close'].shift(1))
nasdaq['True_Range'] = nasdaq[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
nasdaq['ATR_20d'] = nasdaq['True_Range'].rolling(20).mean()

nasdaq['Volatility_10d'] = nasdaq['Close'].pct_change().rolling(10).std()
nasdaq['Volatility_20d'] = nasdaq['Close'].pct_change().rolling(20).std()

# Moving averages and technical indicators
nasdaq['MA_50'] = nasdaq['Close'].rolling(50).mean()
nasdaq['MA_200'] = nasdaq['Close'].rolling(200).mean()
nasdaq['RSI'] = calculate_rsi(nasdaq['Close'])

# Gap and overnight returns
nasdaq['Gap'] = (nasdaq['Open'] - nasdaq['Close'].shift(1)) / nasdaq['Close'].shift(1)
nasdaq['Overnight'] = (nasdaq['Open'] - nasdaq['Close'].shift(1)) / nasdaq['Close'].shift(1)
nasdaq['Intraday'] = (nasdaq['Close'] - nasdaq['Open']) / nasdaq['Open']

# Additional metrics
nasdaq['Daily_Range'] = (nasdaq['High'] - nasdaq['Low']) / nasdaq['Close']
nasdaq['Prev_Close_Positive'] = nasdaq['Close'].pct_change(1) > 0
nasdaq['Volume_Ratio'] = nasdaq['Volume'] / nasdaq['Volume'].rolling(20).mean()

# Calculate MACD
nasdaq['MACD'], nasdaq['MACD_Signal'], nasdaq['MACD_Hist'] = calculate_macd(nasdaq['Close'])

# Step 6: Prepare data for merging
nasdaq = nasdaq.copy()
nasdaq['Datetime'] = nasdaq.index
nasdaq['Date'] = nasdaq['Datetime'].dt.floor('D')
nasdaq['Date'] = nasdaq['Date'].dt.tz_localize(None)
nasdaq = nasdaq.reset_index(drop=True)

# Step 7: Prepare event data for merging
df['Event_Date'] = df['Datetime'].dt.floor('D')
df['Event_Date'] = df['Event_Date'].dt.tz_localize(None)

# Add calendar features to events
df['DayOfWeek'] = df['Event_Date'].dt.dayofweek  # 0=Monday, 4=Friday
df['WeekOfMonth'] = df['Event_Date'].dt.day // 7 + 1
df['Month'] = df['Event_Date'].dt.month
df['IsFriday'] = df['DayOfWeek'] == 4

# Step 8: Merge event data with NASDAQ
merged = pd.merge(df, nasdaq, left_on='Event_Date', right_on='Date', how='left')


def comprehensive_event_analysis(event_data, event_name):
    """Perform comprehensive analysis for a single event type"""

    if len(event_data) == 0:
        return None

    event_data = event_data.dropna(subset=['Return_1d'])
    if len(event_data) == 0:
        return None

    analysis = {}

    # I. Return-Based Metrics
    analysis['🔁 RETURN METRICS'] = {}
    returns_section = analysis['🔁 RETURN METRICS']

    for days in [1, 3, 5, 10, 30]:
        col = f'Return_{days}d'
        if col in event_data.columns:
            returns = event_data[col].dropna() * 100
            returns_section[f'Avg return +{days}d (%)'] = f"{returns.mean():.3f}"
            if days == 1:
                returns_section[f'Std dev +{days}d (%)'] = f"{returns.std():.3f}"
                returns_section[f'Max +{days}d (%)'] = f"{returns.max():.3f}"
                returns_section[f'Min +{days}d (%)'] = f"{returns.min():.3f}"
                returns_section[f'> +1% rate'] = f"{(returns > 1).mean() * 100:.1f}%"
                returns_section[f'< -1% rate'] = f"{(returns < -1).mean() * 100:.1f}%"

    # II. Pre-event Behavior
    analysis['⏪ PRE-EVENT BEHAVIOR'] = {}
    pre_section = analysis['⏪ PRE-EVENT BEHAVIOR']

    for days in [1, 3, 5, 10]:
        col = f'Pre_Return_{days}d'
        if col in event_data.columns:
            pre_returns = event_data[col].dropna() * 100
            pre_section[f'Avg return -{days}d (%)'] = f"{pre_returns.mean():.3f}"

    # III. Calendar Factors
    analysis['📆 CALENDAR FACTORS'] = {}
    calendar_section = analysis['📆 CALENDAR FACTORS']

    # Monthly seasonality - quick version
    monthly_returns = {}
    for month in range(1, 13):
        month_data = event_data[event_data['Month'] == month]
        if len(month_data) > 0:
            month_returns = month_data['Return_1d'].dropna() * 100
            if len(month_returns) > 0:
                monthly_returns[month] = month_returns.mean()

    if monthly_returns:
        best_month = max(monthly_returns, key=monthly_returns.get)
        worst_month = min(monthly_returns, key=monthly_returns.get)
        calendar_section['Best month'] = f"{best_month} ({monthly_returns[best_month]:.2f}%)"
        calendar_section['Worst month'] = f"{worst_month} ({monthly_returns[worst_month]:.2f}%)"

    # First week returns
    first_week = event_data[event_data['WeekOfMonth'] == 1]
    if len(first_week) > 0:
        first_week_returns = first_week['Return_1d'].dropna() * 100
        calendar_section['Avg return first week (%)'] = f"{first_week_returns.mean():.3f}"

    # IV. Volatility & Risk Metrics
    analysis['💥 VOLATILITY & RISK'] = {}
    vol_section = analysis['💥 VOLATILITY & RISK']

    # Range expansion
    daily_ranges = event_data['Daily_Range'].dropna() * 100
    vol_section['Avg daily range (%)'] = f"{daily_ranges.mean():.3f}" if len(daily_ranges) > 0 else "N/A"

    # Realized volatility post-event
    post_vol = event_data['Volatility_20d'].dropna() * 100
    vol_section['Avg 20d volatility (%)'] = f"{post_vol.mean():.3f}" if len(post_vol) > 0 else "N/A"

    # Tail risk
    returns_1d = event_data['Return_1d'].dropna() * 100
    if len(returns_1d) > 0:
        vol_section['% >2σ losses'] = f"{(returns_1d < -2 * returns_1d.std()).mean() * 100:.1f}%"
        vol_section['Max drawdown (%)'] = f"{returns_1d.min():.3f}"
        vol_section['Max gain (%)'] = f"{returns_1d.max():.3f}"

    # V. Technical Reactions
    analysis['📈 TECHNICAL SIGNALS'] = {}
    tech_section = analysis['📈 TECHNICAL SIGNALS']

    # Moving average breaks
    ma_breaks = event_data[event_data['Close'] > event_data['MA_50']]
    tech_section['% above MA50'] = f"{len(ma_breaks) / len(event_data) * 100:.1f}%"

    # MACD signals
    macd_bullish = event_data[event_data['MACD'] > event_data['MACD_Signal']]
    tech_section['% MACD bullish'] = f"{len(macd_bullish) / len(event_data) * 100:.1f}%"

    # RSI levels
    rsi_data = event_data['RSI'].dropna()
    tech_section['Avg RSI'] = f"{rsi_data.mean():.1f}" if len(rsi_data) > 0 else "N/A"

    return analysis


class ScrollableEventAnalyzer:
    def __init__(self, merged_data, df_events):
        self.merged_data = merged_data
        self.df_events = df_events
        self.unique_events = df_events['event'].dropna().unique()
        self.current_event_idx = 0

        # Create main window
        self.root = tk.Tk()
        self.root.title("Comprehensive Event Analysis - Scrollable View")
        self.root.geometry("1400x900")

        # Create main frame with scrollbar
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Event selector
        self.setup_event_selector()

        # Create notebook for tabbed view
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Charts")

        # Analysis tab with scrollable text
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Detailed Analysis")

        self.setup_analysis_tab()
        self.update_display()

    def setup_event_selector(self):
        """Setup event selection controls"""
        selector_frame = ttk.Frame(self.main_frame)
        selector_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(selector_frame, text="Select Event:").pack(side=tk.LEFT, padx=(0, 10))

        self.event_var = tk.StringVar()
        self.event_combo = ttk.Combobox(selector_frame, textvariable=self.event_var,
                                        values=list(self.unique_events), width=50)
        self.event_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.event_combo.bind('<<ComboboxSelected>>', self.on_event_selected)

        ttk.Button(selector_frame, text="Previous", command=self.prev_event).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(selector_frame, text="Next", command=self.next_event).pack(side=tk.LEFT, padx=(0, 10))

        self.event_counter = ttk.Label(selector_frame, text="")
        self.event_counter.pack(side=tk.LEFT)

    def setup_analysis_tab(self):
        """Setup scrollable analysis text tab"""
        # Create scrollable text widget
        text_frame = ttk.Frame(self.analysis_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)

        # Text widget
        self.analysis_text = tk.Text(text_frame, wrap=tk.NONE,
                                     yscrollcommand=v_scrollbar.set,
                                     xscrollcommand=h_scrollbar.set,
                                     font=('Courier', 10))

        # Configure scrollbars
        v_scrollbar.config(command=self.analysis_text.yview)
        h_scrollbar.config(command=self.analysis_text.xview)

        # Pack everything
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_chart(self, event_name, event_data, analysis):
        """Create matplotlib chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        event_data_clean = event_data.dropna(subset=['Return_1d'])
        if len(event_data_clean) == 0:
            ax1.text(0.5, 0.5, f'No data available for\n{event_name}',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            ax1.set_title(f'{event_name}\n(No Data Available)', fontsize=14)
            return fig

        # Sort by date for chronological order
        event_data_clean = event_data_clean.sort_values('Event_Date')
        returns_pct = event_data_clean['Return_1d'] * 100

        # 1. Return bars
        bars = ax1.bar(range(len(returns_pct)), returns_pct,
                       color=['green' if x > 0 else 'red' for x in returns_pct],
                       alpha=0.7, edgecolor='black', linewidth=0.5)

        ax1.set_title(f'{event_name} - 1-Day Returns\n({len(returns_pct)} occurrences)', fontsize=12)
        ax1.set_xlabel('Event Occurrence (#)')
        ax1.set_ylabel('1-Day Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # 2. Distribution histogram
        ax2.hist(returns_pct, bins=min(15, len(returns_pct) // 2 + 1),
                 alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Return Distribution', fontsize=12)
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative returns
        cumulative_returns = (1 + event_data_clean['Return_1d']).cumprod()
        ax3.plot(range(len(cumulative_returns)), (cumulative_returns - 1) * 100,
                 color='blue', linewidth=2, marker='o', markersize=4)
        ax3.set_title('Cumulative Strategy Return', fontsize=12)
        ax3.set_xlabel('Event #')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        # 4. Key metrics summary
        ax4.axis('off')
        if analysis and '🔁 RETURN METRICS' in analysis:
            metrics = analysis['🔁 RETURN METRICS']
            summary_text = "KEY METRICS:\n\n"
            for key, value in list(metrics.items())[:8]:
                summary_text += f"{key}: {value}\n"

            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                     verticalalignment='top', fontsize=10, fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

        plt.tight_layout()
        return fig

    def on_event_selected(self, event=None):
        """Handle event selection from combobox"""
        selected_event = self.event_var.get()
        if selected_event in self.unique_events:
            self.current_event_idx = list(self.unique_events).index(selected_event)
            self.update_display()

    def prev_event(self):
        """Go to previous event"""
        if self.current_event_idx > 0:
            self.current_event_idx -= 1
            self.update_display()

    def next_event(self):
        """Go to next event"""
        if self.current_event_idx < len(self.unique_events) - 1:
            self.current_event_idx += 1
            self.update_display()

    def update_display(self):
        """Update the display with current event data"""
        if self.current_event_idx >= len(self.unique_events):
            return

        current_event = self.unique_events[self.current_event_idx]
        self.event_var.set(current_event)

        # Update counter
        self.event_counter.config(text=f"Event {self.current_event_idx + 1} of {len(self.unique_events)}")

        # Get event data
        event_data = self.merged_data[self.merged_data['event'] == current_event].copy()
        analysis = comprehensive_event_analysis(event_data, current_event)

        # Update chart
        self.update_chart(current_event, event_data, analysis)

        # Update analysis text
        self.update_analysis_text(analysis)

    def update_chart(self, event_name, event_data, analysis):
        """Update the chart tab"""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Create new chart
        fig = self.create_chart(event_name, event_data, analysis)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.chart_frame)
        toolbar.update()

    def update_analysis_text(self, analysis):
        """Update the scrollable analysis text"""
        self.analysis_text.delete(1.0, tk.END)

        if not analysis:
            self.analysis_text.insert(tk.END, "No analysis data available.")
            return

        # Format analysis text
        for section_name, metrics in analysis.items():
            self.analysis_text.insert(tk.END, f"\n{section_name}\n")
            self.analysis_text.insert(tk.END, "=" * 50 + "\n")

            for key, value in metrics.items():
                self.analysis_text.insert(tk.END, f"  • {key:<35}: {value}\n")

            self.analysis_text.insert(tk.END, "\n")

    def run(self):
        """Start the application"""
        self.root.mainloop()


def create_scrollable_event_analyzer():
    """Create and run the scrollable event analyzer"""
    print("Creating scrollable event analyzer...")
    analyzer = ScrollableEventAnalyzer(merged, df)
    analyzer.run()


def save_analysis_as_image(analysis, event_name, output_dir="event_analysis_images"):
    """Save detailed analysis as an image"""
    if not analysis:
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create image
    img_width = 1200
    img_height = 1600
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a better font, fallback to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        section_font = ImageFont.truetype("arial.ttf", 18)
        text_font = ImageFont.truetype("arial.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        section_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    y_pos = 30

    # Title
    title = f"DETAILED ANALYSIS: {event_name}"
    draw.text((50, y_pos), title, fill='black', font=title_font)
    y_pos += 60

    # Draw analysis sections
    for section_name, metrics in analysis.items():
        # Section header
        draw.text((50, y_pos), section_name, fill='blue', font=section_font)
        y_pos += 30

        # Draw line under section
        draw.line([(50, y_pos), (img_width - 50, y_pos)], fill='gray', width=2)
        y_pos += 20

        # Metrics
        for key, value in metrics.items():
            text = f"  • {key}: {value}"
            draw.text((70, y_pos), text, fill='black', font=text_font)
            y_pos += 25

        y_pos += 20

        # Check if we need to break (prevent overflow)
        if y_pos > img_height - 100:
            break

    # Save image
    safe_event_name = event_name.replace('/', '_').replace('\\', '_')
    filename = f"{safe_event_name}_analysis.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    return filepath


def save_all_events_as_images():
    """Save charts and analysis for all events as images"""
    output_dir = "event_analysis_images"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving analysis images to: {os.path.abspath(output_dir)}")

    saved_count = 0

    for i, event_name in enumerate(merged['event'].dropna().unique()):
        print(f"Processing {i + 1}/{len(merged['event'].dropna().unique())}: {event_name}")

        # Get event data
        event_data = merged[merged['event'] == event_name].copy()
        if len(event_data) == 0:
            continue

        # Get analysis
        analysis = comprehensive_event_analysis(event_data, event_name)
        if not analysis:
            continue

        try:
            # Create and save chart
            fig = plt.figure(figsize=(16, 12))

            # Create subplots
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Full width for returns
            ax2 = fig.add_subplot(gs[1, 0])  # Distribution
            ax3 = fig.add_subplot(gs[1, 1])  # Cumulative
            ax4 = fig.add_subplot(gs[2, :])  # Full width for analysis text

            event_data_clean = event_data.dropna(subset=['Return_1d'])

            if len(event_data_clean) > 0:
                # Sort by date
                event_data_clean = event_data_clean.sort_values('Event_Date')
                returns_pct = event_data_clean['Return_1d'] * 100

                # 1. Return bars
                bars = ax1.bar(range(len(returns_pct)), returns_pct,
                               color=['green' if x > 0 else 'red' for x in returns_pct],
                               alpha=0.7, edgecolor='black', linewidth=0.5)
                ax1.set_title(f'{event_name} - 1-Day Returns ({len(returns_pct)} occurrences)',
                              fontsize=14, pad=20)
                ax1.set_xlabel('Event Occurrence (#)')
                ax1.set_ylabel('1-Day Return (%)')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

                # 2. Distribution
                ax2.hist(returns_pct, bins=min(15, len(returns_pct) // 2 + 1),
                         alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title('Return Distribution', fontsize=12)
                ax2.set_xlabel('Return (%)')
                ax2.set_ylabel('Frequency')
                ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax2.grid(True, alpha=0.3)

                # 3. Cumulative returns
                cumulative_returns = (1 + event_data_clean['Return_1d']).cumprod()
                ax3.plot(range(len(cumulative_returns)), (cumulative_returns - 1) * 100,
                         color='blue', linewidth=2, marker='o', markersize=4)
                ax3.set_title('Cumulative Strategy Return', fontsize=12)
                ax3.set_xlabel('Event #')
                ax3.set_ylabel('Cumulative Return (%)')
                ax3.grid(True, alpha=0.3)
                ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            else:
                ax1.text(0.5, 0.5, f'No data available for\n{event_name}',
                         ha='center', va='center', transform=ax1.transAxes, fontsize=16)

            # 4. Analysis summary on chart
            ax4.axis('off')
            if analysis:
                summary_text = "KEY METRICS:\n\n"

                # Get key metrics from different sections
                sections_to_show = ['🔁 RETURN METRICS', '⏪ PRE-EVENT BEHAVIOR', '💥 VOLATILITY & RISK']

                for section_name in sections_to_show:
                    if section_name in analysis:
                        summary_text += f"{section_name}\n"
                        metrics = analysis[section_name]

                        # Show first 6 metrics from each section
                        for i, (key, value) in enumerate(list(metrics.items())[:6]):
                            summary_text += f"  • {key}: {value}\n"
                        summary_text += "\n"

                ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                         verticalalignment='top', fontsize=10, fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

            # Save chart
            safe_event_name = event_name.replace('/', '_').replace('\\', '_')
            chart_filename = f"{safe_event_name}_chart.png"
            chart_filepath = os.path.join(output_dir, chart_filename)

            plt.savefig(chart_filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            # Save detailed analysis as image
            analysis_filepath = save_analysis_as_image(analysis, event_name, output_dir)

            saved_count += 1

        except Exception as e:
            print(f"Error processing {event_name}: {str(e)}")
            plt.close('all')  # Clean up any open figures
            continue

    print(f"\n✅ Successfully saved {saved_count} events to: {os.path.abspath(output_dir)}")
    print(f"   📈 Chart images: *_chart.png")
    print(f"   📊 Analysis images: *_analysis.png")


# Add export button to the existing GUI
def add_export_functionality_to_analyzer():
    """Add export functionality to the existing ScrollableEventAnalyzer class"""

    # Add this method to the ScrollableEventAnalyzer class
    def export_current_event(self):
        """Export current event as images"""
        if self.current_event_idx >= len(self.unique_events):
            return

        current_event = self.unique_events[self.current_event_idx]
        event_data = self.merged_data[self.merged_data['event'] == current_event].copy()
        analysis = comprehensive_event_analysis(event_data, current_event)

        output_dir = "event_analysis_images"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save chart
            fig = self.create_chart(current_event, event_data, analysis)
            safe_event_name = current_event.replace('/', '_').replace('\\', '_')
            chart_filename = f"{safe_event_name}_chart.png"
            chart_filepath = os.path.join(output_dir, chart_filename)

            fig.savefig(chart_filepath, dpi=300, bbox_inches='tight', facecolor='white')

            # Save analysis
            analysis_filepath = save_analysis_as_image(analysis, current_event, output_dir)

            print(f"✅ Exported {current_event}:")
            print(f"   📈 Chart: {chart_filepath}")
            print(f"   📊 Analysis: {analysis_filepath}")

        except Exception as e:
            print(f"❌ Error exporting {current_event}: {str(e)}")

    # Add the method to the class
    ScrollableEventAnalyzer.export_current_event = export_current_event

    # Modify the setup_event_selector method to include export buttons
    original_setup = ScrollableEventAnalyzer.setup_event_selector

    def enhanced_setup_event_selector(self):
        original_setup(self)

        # Add export buttons
        export_frame = ttk.Frame(self.main_frame)
        export_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(export_frame, text="📸 Export Current Event",
                   command=self.export_current_event).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(export_frame, text="📁 Export All Events",
                   command=lambda: save_all_events_as_images()).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(export_frame, text="(Images saved to 'event_analysis_images' folder)",
                  foreground='gray').pack(side=tk.LEFT, padx=(10, 0))

    ScrollableEventAnalyzer.setup_event_selector = enhanced_setup_event_selector


# Run the enhancements
add_export_functionality_to_analyzer()

# Execute the scrollable analyzer
print("Starting scrollable event analysis...")
create_scrollable_event_analyzer()

print("\n" + "=" * 80)
print("SCROLLABLE EVENT ANALYSIS COMPLETED")
print("=" * 80)

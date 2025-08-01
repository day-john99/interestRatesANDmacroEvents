import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download bond yields and NASDAQ data
us10y = yf.download("^TNX", start="2000-01-01")
us3m = yf.download("^IRX", start="2000-01-01")
nasdaq = yf.download("^IXIC", start="2000-01-01")

# Step 2: Flatten multi-level columns
us10y.columns = us10y.columns.get_level_values(0)
us3m.columns = us3m.columns.get_level_values(0)
nasdaq.columns = nasdaq.columns.get_level_values(0)

# Step 3: Extract 'Close' prices
data = pd.DataFrame({
    "10Y Yield": us10y['Close'],
    "3M Yield": us3m['Close'],
    "NASDAQ": nasdaq['Close']
})

# Step 4: Clean and process
data.dropna(inplace=True)
data["Yield Spread"] = data["10Y Yield"] - data["3M Yield"]
data["Inverted"] = data["Yield Spread"] < 0

# Step 5: Plot
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot yield spread
ax1.plot(data.index, data["Yield Spread"], label="10Y - 3M Yield Spread", color='blue')
ax1.axhline(0, color='gray', linestyle='--')
ax1.set_ylabel("Yield Spread (%)")
ax1.set_title("10Y vs 3M Yield Spread & NASDAQ Composite - Inversion Signals")
ax1.legend(loc='upper left')

# Step 6: Highlight inversion periods
inversion_start = None
for i in range(1, len(data)):
    if data["Inverted"].iloc[i] and not data["Inverted"].iloc[i - 1]:
        inversion_start = data.index[i]
    elif not data["Inverted"].iloc[i] and data["Inverted"].iloc[i - 1] and inversion_start:
        inversion_end = data.index[i]
        ax1.axvspan(inversion_start, inversion_end, color='red', alpha=0.3)
        inversion_start = None

# Step 7: Plot NASDAQ on secondary axis
ax2 = ax1.twinx()
ax2.plot(data.index, data["NASDAQ"], label="NASDAQ Composite", color='green', alpha=0.6)
ax2.set_ylabel("NASDAQ Level")

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.show()

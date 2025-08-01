'''

Great! You’ve created a Granger causality matrix, which is like asking:

“Does one economic indicator help predict another?”

Let’s break it down simply 👇

🧠 HOW TO READ THE CHART:
Rows: The cause — “Does this indicator help predict...”

Columns: The effect — “…this indicator?”

Cell value = p-value: Lower = stronger predictive signal

✅ p < 0.05 → statistically meaningful (highlighted in dark blue)

❌ p > 0.05 → weak or no predictive power
'''


import pandas as pd
from fredapi import Fred
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from dotenv import load_dotenv
import os


load_dotenv()

# FRED API key (replace with your own key)
FRED_API_KEY = os.getenv("FRED_API_KEY")  # Replace with your FRED API key
fred = Fred(api_key=FRED_API_KEY)

# Date range
start_date = "2000-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# Mapping your indicators to FRED series codes
indicator_map = {
    "Federal Funds Rate": "FEDFUNDS",
    "Unemployment Rate": "UNRATE",
    "10-Year Treasury Rate": "GS10",
    "3-Month Treasury Rate": "GS3M",
    "VIX (Market Volatility)": "VIXCLS",
    "Consumer Confidence": "UMCSENT",
    "CPI Inflation": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "GDP Growth": "A191RL1Q225SBEA",
    "Corporate Bond Spreads": "BAA10Y",
    "M2 Money Supply": "M2SL",
    "Real Estate Prices": "CSUSHPISA"
}

# Fetch data
all_data = {}
for name, code in indicator_map.items():
    try:
        df = fred.get_series(code, start_date, end_date)
        all_data[name] = df
    except Exception as e:
        print(f"Error fetching {name} ({code}): {e}")

# Combine into a DataFrame
df_all = pd.DataFrame(all_data)

# Resample to monthly to align different frequencies
df_monthly = df_all.resample("M").last()

# Drop rows with missing values
df_clean = df_monthly.dropna()

# ----------------------------- #
# 📈 Granger Causality Function
# ----------------------------- #
def grangers_causality_matrix(data, variables, maxlag=3, verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            if r != c:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)]
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
            else:
                df.loc[r, c] = np.nan
    return df

# Run Granger Causality Matrix
variables = df_clean.columns.tolist()
granger_matrix = grangers_causality_matrix(df_clean, variables, maxlag=3)

# Show heatmap of Granger p-values
plt.figure(figsize=(12, 8))
sns.heatmap(granger_matrix, annot=True, cmap="coolwarm", fmt=".2g", cbar_kws={'label': 'p-value'})
plt.title("Granger Causality - Does Row Cause Column?")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

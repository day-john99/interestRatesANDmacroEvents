# 📈 Macro Events & NASDAQ Analysis

This project quantitatively analyzes how **U.S. macroeconomic events**—like CPI, NFP, JOLTS, and Interest Rate decisions—affect the **NASDAQ index**. It investigates **next-day returns**, **pivot-based directional bias**, and **crash risk prediction** using methods like Granger Causality and bond yield inversion.

Ideal for:
- **Quantitative analysts** building macro-driven strategies  
- **Traders** interested in event-based edge  
- **HFT desks** seeking systematic alpha triggers around high-impact events

---

## 📊 Key Features

- 📅 **Macro Event Analysis**
  - Returns after CPI, NFP, JOLTS, FOMC events
  - Impact categorized by better/worse than forecast
  - Weekday seasonality of events

- 🧮 **Return-Based Strategies**
  - Daily and delta-neutral strategy simulation
  - Pivot-level vs. open-based directional bias

- 📉 **Crash Risk Estimation**
  - Bond yield inversion vs NASDAQ drops
  - Extreme macro event filters
  - Granger causality analysis

---

## 🗂️ Project Structure

.
├── event_analysis_images/ # Visual outputs (charts/plots)
├── cleaned_economic_events.csv # Cleaned macroeconomic event dataset
├── extreme_economic_events.csv # Filtered high-impact macro events
├── nasdaq_daily_2000_to_now.csv # NASDAQ OHLCV data
├── nasdaq_event_analysis.csv # Main output: event-wise NASDAQ performance
├── correlation_summary.json # Summary of correlation analyses
├── crash_risk_report.json # Output of crash risk model
│
├── MacroEventsAnalysisGeneral.py # General macro impact analysis
├── MacroEventsAnalysisWRTPrvDayOpen.py # Return analysis wrt previous day open
├── MacroEventsAnalysisWRTEventsPrvDayDeltaNeutralityTrades.py # Strategy simulation
├── bondYieldInversionForCrashes.py # Crash prediction via bond yields
├── gragnerCasualityWRTeconomicEvents.py # Granger causality analysis
│
├── requirements.txt # Required Python packages
├── .gitignore


---

## ⚙️ Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/day-john99/interestRatesANDmacroEvents.git
cd interestRatesANDmacroEvents
2. Install Requirements
bash
pip install -r requirements.txt
3. Run Any Analysis Script
Example:

bash

python MacroEventsAnalysisWRTPrvDayOpen.py
✅ Output Highlights
nasdaq_event_analysis.csv: Return statistics post-events

crash_risk_report.json: Bond yield inversion crash warnings

event_analysis_images/: Plots and charts of analysis

Strategy simulation files: Return vs prior day delta neutral setups

📌 Use Case
This analysis can help you:

Design event-based trades using historical edge

Pre-position for major releases with delta-neutral strategies

Estimate macro-driven crash probabilities

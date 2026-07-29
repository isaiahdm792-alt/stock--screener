# STEP 1: Fundamentals Scanner — starter script
# Paste this into a Google Colab cell and run it.
# First run: install yfinance (only needed once per Colab session)

# !pip install yfinance --quiet

import yfinance as yf
import pandas as pd
import time

# --- Start small: 15 well-known tickers to test the pipeline first ---
# Once this works cleanly, swap in the full S&P 500 list (step below).
test_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AVGO", "PLTR", "CRWD",
    "SHOP", "RKLB", "AMD", "NFLX", "JPM"
]

def get_fundamentals(ticker):
    """Pull key fundamental ratios for one ticker. Returns a dict or None on failure."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE")
        ps = info.get("priceToSalesTrailing12Months")
        pb = info.get("priceToBook")
        peg = info.get("pegRatio")
        fcf = info.get("freeCashflow")
        earnings_growth = info.get("earningsGrowth")
        sector = info.get("sector")

        return {
            "ticker": ticker,
            "sector": sector,
            "pe_ratio": pe,
            "ps_ratio": ps,
            "pb_ratio": pb,
            "peg_ratio": peg,
            "free_cash_flow": fcf,
            "earnings_growth": earnings_growth,
        }
    except Exception as e:
        print(f"  [!] Failed on {ticker}: {e}")
        return None

# --- Run the scan ---
results = []
for t in test_tickers:
    print(f"Pulling {t}...")
    data = get_fundamentals(t)
    if data:
        results.append(data)
    time.sleep(0.5)  # be polite to the API, avoid rate-limit issues

df = pd.DataFrame(results)

# --- Basic cleanup: drop rows missing the core ratios ---
df_clean = df.dropna(subset=["pe_ratio", "ps_ratio"])

print("\n--- Results ---")
print(df_clean.sort_values("pe_ratio").to_string(index=False))

# --- Save to CSV so you can commit it to GitHub / inspect later ---
df_clean.to_csv("fundamentals_snapshot.csv", index=False)
print("\nSaved to fundamentals_snapshot.csv")

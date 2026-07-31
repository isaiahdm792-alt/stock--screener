# daily_scan.py
# Combined pipeline: fundamentals -> technicals -> scoring -> sentiment -> insider -> macro
# Designed to run non-interactively via GitHub Actions (no Colab-specific code).

import os
import time
import requests
import pandas as pd
from io import StringIO

# --- API keys come from environment variables (set as GitHub Secrets), not Colab userdata ---
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")

SEC_HEADERS = {"User-Agent": "Isaiah - personal stock research project - davismooreisaiah@gmail.com"}


# ============================================================
# STEP 1: S&P 500 ticker list
# ============================================================
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    tickers = tables[0]["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]


# ============================================================
# STEP 1b: Fundamentals scan
# ============================================================
import yfinance as yf

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "sector": info.get("sector"),
            "pe_ratio": info.get("trailingPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "peg_ratio": info.get("pegRatio"),
            "free_cash_flow": info.get("freeCashflow"),
            "earnings_growth": info.get("earningsGrowth"),
        }
    except Exception as e:
        print(f"  [!] Fundamentals failed on {ticker}: {e}")
        return None


# ============================================================
# STEP 2: Technical indicators (manual calc, no pandas-ta)
# ============================================================
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def get_technicals(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty or len(hist) < 200:
            return None

        hist["sma50"] = hist["Close"].rolling(window=50).mean()
        hist["sma200"] = hist["Close"].rolling(window=200).mean()
        hist["rsi"] = calculate_rsi(hist["Close"])
        hist["macd"], hist["macd_signal"] = calculate_macd(hist["Close"])

        avg_volume_30d = hist["Volume"].tail(30).mean()
        latest = hist.iloc[-1]

        return {
            "ticker": ticker,
            "price": latest["Close"],
            "above_sma50": latest["Close"] > latest["sma50"],
            "above_sma200": latest["Close"] > latest["sma200"],
            "rsi": latest["rsi"],
            "macd_bullish": latest["macd"] > latest["macd_signal"],
            "volume_today": latest["Volume"],
            "avg_volume_30d": avg_volume_30d,
            "volume_spike": latest["Volume"] > (avg_volume_30d * 1.5),
        }
    except Exception as e:
        print(f"  [!] Technicals failed on {ticker}: {e}")
        return None


def normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 100


# ============================================================
# STEP 5: Sentiment (NewsAPI + FinBERT), top candidates only
# ============================================================
def get_headlines(ticker, api_key, page_size=10):
    url = "https://newsapi.org/v2/everything"
    params = {"q": ticker, "language": "en", "sortBy": "publishedAt", "pageSize": page_size, "apiKey": api_key}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") != "ok":
            print(f"  [!] NewsAPI error for {ticker}: {data.get('message')}")
            return []
        return [a["title"] for a in data.get("articles", []) if a.get("title")]
    except Exception as e:
        print(f"  [!] News fetch failed for {ticker}: {e}")
        return []

def score_sentiment(headlines, sentiment_pipeline):
    if not headlines:
        return None
    results = sentiment_pipeline(headlines)
    total = 0
    for r in results:
        if r["label"] == "positive":
            total += r["score"]
        elif r["label"] == "negative":
            total -= r["score"]
    return total / len(results)


# ============================================================
# STEP 6a: Insider filings (SEC EDGAR) - activity count, not yet buy/sell direction
# ============================================================
def get_cik(ticker, ticker_map):
    return ticker_map.get(ticker)

def get_insider_filings(ticker, cik, months_back=3):
    if not cik:
        return {"ticker": ticker, "recent_form4_filings": None}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=10)
        data = response.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
        count = sum(1 for f, d in zip(forms, filing_dates) if f == "4" and pd.Timestamp(d) >= cutoff)
        return {"ticker": ticker, "recent_form4_filings": count}
    except Exception as e:
        print(f"  [!] Insider filing pull failed for {ticker}: {e}")
        return {"ticker": ticker, "recent_form4_filings": None}


# ============================================================
# STEP 6b: Macro (FRED)
# ============================================================
def get_fred_series(series_id, api_key, limit=1):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "sort_order": "desc", "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    return response.json().get("observations", [])


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=== Step 1: Pulling S&P 500 ticker list ===")
    sp500_tickers = get_sp500_tickers()
    print(f"Pulled {len(sp500_tickers)} tickers.")

    print("\n=== Step 1b: Fundamentals scan ===")
    fund_results = []
    for t in sp500_tickers:
        data = get_fundamentals(t)
        if data:
            fund_results.append(data)
        time.sleep(0.5)
    df_fund = pd.DataFrame(fund_results).dropna(subset=["pe_ratio", "ps_ratio"])
    print(f"Fundamentals collected for {len(df_fund)} tickers.")

    sector_avg_pe = df_fund.groupby("sector")["pe_ratio"].transform("mean")
    df_fund["pe_vs_sector"] = df_fund["pe_ratio"] - sector_avg_pe
    df_fund.to_csv("fundamentals_snapshot.csv", index=False)

    print("\n=== Step 2: Technical indicators ===")
    tech_results = []
    for t in sp500_tickers:
        data = get_technicals(t)
        if data:
            tech_results.append(data)
        time.sleep(0.5)
    df_tech = pd.DataFrame(tech_results)
    df_tech.to_csv("technicals_snapshot.csv", index=False)
    print(f"Technicals collected for {len(df_tech)} tickers.")

    print("\n=== Merging + scoring ===")
    df_combined = df_fund.merge(df_tech, on="ticker", how="inner")
    df_combined["fundamentals_score"] = normalize(1 / df_combined["peg_ratio"].clip(lower=0.1))
    df_combined["technicals_score"] = (
        df_combined["above_sma50"].astype(int) * 25
        + df_combined["above_sma200"].astype(int) * 25
        + df_combined["macd_bullish"].astype(int) * 25
        + df_combined["volume_spike"].astype(int) * 25
    )
    df_combined["score"] = df_combined["fundamentals_score"] * 0.6 + df_combined["technicals_score"] * 0.4
    df_combined = df_combined.sort_values("score", ascending=False)
    print(f"Merged and scored {len(df_combined)} tickers.")

    print("\n=== Step 5: Sentiment (top 30 candidates) ===")
    top_candidates = df_combined.head(30)["ticker"].tolist()
    df_final = df_combined.copy()

    if NEWSAPI_KEY:
        from transformers import pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

        sentiment_results = []
        for t in top_candidates:
            headlines = get_headlines(t, NEWSAPI_KEY)
            sent_score = score_sentiment(headlines, sentiment_pipeline)
            sentiment_results.append({"ticker": t, "headline_count": len(headlines), "sentiment_score": sent_score})
        df_sentiment = pd.DataFrame(sentiment_results)
        df_sentiment.to_csv("sentiment_snapshot.csv", index=False)

        skipped = df_sentiment[df_sentiment["sentiment_score"].isna()]["ticker"].tolist()
        if skipped:
            print(f"[!] {len(skipped)} tickers had no sentiment data (rate limit or no headlines): {skipped}")

        df_final = df_final.merge(df_sentiment, on="ticker", how="left")
        df_final["sentiment_norm"] = normalize(df_final["sentiment_score"].fillna(0))
        df_final["score"] = (
            df_final["fundamentals_score"] * 0.45
            + df_final["technicals_score"] * 0.35
            + df_final["sentiment_norm"].fillna(50) * 0.20
        )
    else:
        print("NEWSAPI_KEY not set - skipping sentiment layer.")

    print("\n=== Step 6a: Insider filings (top 50 candidates) ===")
    lookup_url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(lookup_url, headers=SEC_HEADERS, timeout=10)
        raw_map = response.json()
        ticker_map = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in raw_map.values()}
    except Exception as e:
        print(f"  [!] Could not load ticker->CIK map: {e}")
        ticker_map = {}

    insider_results = []
    for t in top_candidates:
        cik = get_cik(t, ticker_map)
        insider_results.append(get_insider_filings(t, cik))
        time.sleep(0.3)
    df_insider = pd.DataFrame(insider_results)
    df_insider.to_csv("insider_snapshot.csv", index=False)
    print("Insider filings saved (activity count only - buy/sell direction not yet parsed, parked for a future step).")

    print("\n=== Step 6b: Macro (FRED) ===")
    if FRED_API_KEY:
        fed_funds = get_fred_series("FEDFUNDS", FRED_API_KEY)
        cpi = get_fred_series("CPIAUCSL", FRED_API_KEY, limit=13)
        unemployment = get_fred_series("UNRATE", FRED_API_KEY)

        current_fed_funds = float(fed_funds[0]["value"])
        current_unemployment = float(unemployment[0]["value"])
        cpi_now = float(cpi[0]["value"])
        cpi_year_ago = float(cpi[-1]["value"])
        inflation_yoy = ((cpi_now - cpi_year_ago) / cpi_year_ago) * 100

        rate_regime = "high_rates" if current_fed_funds > 4.5 else ("moderate_rates" if current_fed_funds > 2 else "low_rates")

        pd.DataFrame([{
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "fed_funds_rate": current_fed_funds,
            "unemployment_rate": current_unemployment,
            "inflation_yoy": inflation_yoy,
            "rate_regime": rate_regime,
        }]).to_csv("macro_snapshot.csv", index=False)

        if rate_regime in ["high_rates", "moderate_rates"]:
            df_final["macro_adjustment"] = df_final["fundamentals_score"].apply(lambda x: 3 if x > 60 else (-3 if x < 30 else 0))
        else:
            df_final["macro_adjustment"] = df_final["technicals_score"].apply(lambda x: 3 if x > 60 else 0)
        df_final["score"] = df_final["score"] + df_final["macro_adjustment"]
        print(f"Applied macro regime: {rate_regime}")
    else:
        print("FRED_API_KEY not set - skipping macro layer.")

    df_final = df_final.sort_values("score", ascending=False)
    df_final.to_csv("scored_snapshot.csv", index=False)

    print("\n=== Done ===")
    print(f"Final rows: {len(df_final)}")
    print(f"Score range: {df_final['score'].min():.1f} to {df_final['score'].max():.1f}")
    print(f"Columns: {df_final.columns.tolist()}")


if __name__ == "__main__":
    main()

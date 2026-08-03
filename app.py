import streamlit as st
import pandas as pd

st.set_page_config(page_title="Stock Screener", layout="wide")
st.title("Stock Screener")

df = pd.read_csv("scored_snapshot.csv")

model = st.selectbox("Screening model", ["Overall", "Value", "Growth", "Momentum"])
score_column = {"Overall": "score", "Value": "value_score", "Growth": "growth_composite", "Momentum": "momentum_score"}[model]

# --- Filters ---
col1, col2, col3 = st.columns(3)
with col1:
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Sector", sectors)
with col2:
    min_score = st.slider("Minimum score", 0, int(df[score_column].max()), 0)
with col3:
    ticker_search = st.text_input("Search ticker")

filtered = df.copy()
if sector_filter != "All":
    filtered = filtered[filtered["sector"] == sector_filter]
filtered = filtered[filtered[score_column] >= min_score]
if ticker_search:
    filtered = filtered[filtered["ticker"].str.contains(ticker_search.upper())]

filtered = filtered.sort_values(score_column, ascending=False)

# --- Main layout: table on the left, detail panel on the right ---
left, right = st.columns([2, 1])

with left:
    st.subheader(f"Ranked stocks ({len(filtered)})")
    st.dataframe(
        filtered[["ticker", score_column, "sector", "fundamentals_score", "technicals_score"]],
        use_container_width=True,
        height=500,
    )

with right:
    st.subheader("Stock detail")
    selected = st.selectbox("Select a ticker to inspect", filtered["ticker"].tolist())

    if selected:
        row = filtered[filtered["ticker"] == selected].iloc[0]

        st.metric(f"{model} score", f"{row[score_column]:.1f}")

        st.write("**Fundamentals**")
        st.progress(min(int(row["fundamentals_score"]), 100) / 100)

        st.write("**Technicals**")
        st.progress(min(int(row["technicals_score"]), 100) / 100)

        if "sentiment_norm" in row:
            st.write("**Sentiment**")
            st.progress(min(int(row["sentiment_norm"]), 100) / 100)

        st.write("---")
        st.write(f"Sector: {row['sector']}")
        st.write(f"PEG ratio: {row.get('peg_ratio', 'N/A')}")
        st.write(f"Above 200-day SMA: {row.get('above_sma200', 'N/A')}")

        st.write("---")
        st.write(f"**Confidence:** {row.get('confidence_pct', 0):.0f}% ({int(row.get('sources_present', 0))}/4 data sources)")
        st.write(f"**AI Summary:** {row.get('ai_summary', 'Not available')}")

st.write("---")
st.subheader("Compare stocks")

compare_tickers = st.multiselect("Select 2-3 tickers to compare", df["ticker"].tolist())

if len(compare_tickers) >= 2:
    compare_df = df[df["ticker"].isin(compare_tickers)].set_index("ticker")

    comparison_table = compare_df[[
        "score", "value_score", "growth_composite", "momentum_score",
        "fundamentals_score", "technicals_score", "sentiment_norm",
        "peg_ratio", "confidence_pct"
    ]].T  # transpose so tickers become columns, metrics become rows

    st.dataframe(comparison_table, use_container_width=True)
elif len(compare_tickers) == 1:
    st.info("Select at least one more ticker to compare.")      

import streamlit as st
import pandas as pd

st.title("Stock Screener")

df = pd.read_csv("scored_snapshot.csv")

st.subheader("Top ranked stocks")
st.dataframe(df.sort_values("score", ascending=False).head(30))

ticker_filter = st.text_input("Search a ticker")
if ticker_filter:
    st.write(df[df["ticker"].str.contains(ticker_filter.upper())])

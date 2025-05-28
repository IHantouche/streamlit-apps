
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# ---------------------- Helper Functions ----------------------

def fetch_data(ticker, period="1y", interval="1d"):
    """Fetches historical data for a given ticker."""
    return yf.Ticker(ticker).history(period=period, interval=interval)


def compute_indicators(df, ma_windows=[20,50], bb_window=20, bb_std=2, rsi_period=14):
    """Compute moving averages, Bollinger Bands, and RSI."""
    data = df.copy()
    # Moving Averages
    for w in ma_windows:
        data[f"MA_{w}"] = data['Close'].rolling(window=w).mean()
    # Bollinger Bands
    rolling = data['Close'].rolling(window=bb_window)
    data['BB_MID'] = rolling.mean()
    data['BB_STD'] = rolling.std()
    data['BB_UPPER'] = data['BB_MID'] + bb_std * data['BB_STD']
    data['BB_LOWER'] = data['BB_MID'] - bb_std * data['BB_STD']
    # RSI
    delta = data['Close'].diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    ma_up = up.rolling(window=rsi_period).mean()
    ma_down = down.rolling(window=rsi_period).mean()
    rs = ma_up / ma_down
    data['RSI'] = 100 - (100/(1+rs))
    return data.dropna()


def plot_candlestick(df, indicators, title, colors):
    """Generate interactive candlestick chart with overlays."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=(title, 'RSI'),
                        row_width=[0.2, 0.7])
    # Candlestick
    fig.add_trace(
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                       low=df['Low'], close=df['Close'], name='Price'), row=1, col=1
    )
    # Moving Averages
    for ma in [c for c in df.columns if c.startswith('MA_')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines', name=ma), row=1, col=1)
    # Bollinger Bands
    for band in ['BB_UPPER', 'BB_LOWER']:
        fig.add_trace(go.Scatter(x=df.index, y=df[band],
                                 line=dict(dash='dash'), name=band), row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color=colors['rsi']), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    return fig


def compute_portfolio(df, weights):
    """Calculate portfolio value and allocations."""
    returns = df['Close'].pct_change().dropna()
    norm = (1 + returns).cumprod()
    alloc = norm.mul(weights, axis=1)
    port_val = alloc.sum(axis=1)
    return port_val, alloc


def compute_correlations(df):
    """Return correlation matrix of closing prices."""
    return df['Close'].corr()

# ---------------------- Streamlit App ----------------------

def main():
    st.title("📈 Stock Market Visualizer")
    st.sidebar.header("Configuration")

    # User inputs
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="AAPL")
    period = st.sidebar.selectbox("Period", ['1mo','3mo','6mo','1y','2y','5y','max'], index=3)
    interval = st.sidebar.selectbox("Interval", ['1d','1wk','1mo'], index=0)
    ma_windows = st.sidebar.multiselect("Moving Averages", [10,20,50,100,200], default=[20,50])
    bb_window = st.sidebar.slider("BB Window", 10, 50, 20)
    bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)
    rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
    rsi_color = st.sidebar.color_picker("RSI Line Color", '#636EFA')

    # Fetch and compute
    data = fetch_data(ticker, period, interval)
    df = compute_indicators(data, ma_windows, bb_window, bb_std, rsi_period)

    # Plot
    fig = plot_candlestick(df, {}, title=f"{ticker} Price Chart", colors={'rsi': rsi_color})
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio upload
    st.sidebar.header("Portfolio Tracking")
    uploaded = st.sidebar.file_uploader("Upload portfolio CSV/Excel", type=['csv','xlsx'])
    if uploaded:
        port_df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        tickers = port_df['Ticker'].tolist()
        weights = port_df.set_index('Ticker')['Weight']/100
        price_data = pd.concat([fetch_data(t, period, interval)['Close'].rename(t) for t in tickers], axis=1)
        port_val, alloc = compute_portfolio(price_data, weights)
        st.line_chart(port_val, height=200)
        st.write("#### Correlation Matrix")
        st.dataframe(compute_correlations(price_data))

    # Export and settings
    if st.button("Export Chart as HTML"):
        html = fig.to_html()
        st.download_button("Download HTML", html, file_name=f"{ticker}_chart.html", mime='text/html')
    if st.button("Export Chart as PNG"):
        png = fig.to_image(format='png')
        st.download_button("Download PNG", png, file_name=f"{ticker}_chart.png", mime='image/png')

    # Save / load config
    st.sidebar.header("User Settings")
    if st.sidebar.button("Save Configuration"):
        cfg = {
            'ticker': ticker,
            'period': period,
            'interval': interval,
            'ma_windows': ma_windows,
            'bb_window': bb_window,
            'bb_std': bb_std,
            'rsi_period': rsi_period,
            'rsi_color': rsi_color
        }
        st.sidebar.download_button("Download Config JSON", data=json.dumps(cfg), file_name="config.json")
    uploaded_cfg = st.sidebar.file_uploader("Load Config JSON", type=['json'], key='cfg')
    if uploaded_cfg:
        cfg = json.load(uploaded_cfg)
        st.experimental_rerun()

if __name__ == "__main__":
    main()

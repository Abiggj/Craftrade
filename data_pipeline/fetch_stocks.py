"""
Multi-Decade Automated Stock & Benchmark Data Collector for CrafTrade.
Fetches daily OHLCV and volatility metrics from 2000 to present across:
- Macro & Benchmark Indices (Nifty 50, BSE Sensex, India VIX, Nifty Bank, Nifty IT)
- Large-Cap Bellwethers across Sectors (IT, Banking, PSU, Energy, Conglomerates)
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Comprehensive tickers including Macro Benchmarks and Sector Leaders
TICKERS = {
    # Benchmark & Macro Indices
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "INDIAVIX": "^INDIAVIX",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    
    # IT Bellwethers
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    
    # Banking & Financials (Private & PSU)
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "IDBI": "IDBI.NS",
    
    # Energy, Conglomerate & Heavyweights
    "RELIANCE": "RELIANCE.NS",
    "LT": "LT.NS",
    "TATAMOTORS": "TATAMOTORS.NS"
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stocks")

def fetch_stock_data(start_date="2000-01-01", end_date=None):
    """
    Downloads multi-decade historical data and saves both individual series
    and a unified multi-asset matrix.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"[*] Fetching multi-decade market data from {start_date} to {end_date}...")
    all_series = []

    for symbol, yahoo_ticker in TICKERS.items():
        print(f"    -> Pulling {symbol} [{yahoo_ticker}]...")
        try:
            ticker_obj = yf.Ticker(yahoo_ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                print(f"       [!] Warning: No data returned for {yahoo_ticker}")
                continue

            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

            # Retain core columns
            cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[c for c in cols if c in df.columns]]
            df.columns = ['Date'] + [f"{col}_{symbol}" for col in df.columns if col != 'Date']

            # Calculate daily percentage return
            close_col = f"Close_{symbol}"
            open_col = f"Open_{symbol}"
            if close_col in df.columns and open_col in df.columns:
                df[f'Return_Pct_{symbol}'] = ((df[close_col] - df[open_col]) / df[open_col]) * 100

            # Save individual file
            individual_file = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
            df.to_csv(individual_file, index=False)
            all_series.append(df)

        except Exception as e:
            print(f"       [!] Error fetching {yahoo_ticker}: {e}")

    if not all_series:
        print("[!] No data fetched.")
        return None

    print("[*] Merging all assets into consolidated master matrix...")
    master_df = all_series[0]
    for df in all_series[1:]:
        master_df = pd.merge(master_df, df, on='Date', how='outer')

    master_df = master_df.sort_values(by='Date').reset_index(drop=True)
    master_file = os.path.join(OUTPUT_DIR, "master_stocks.csv")
    master_df.to_csv(master_file, index=False)

    print(f"[+] Multi-decade data saved to {master_file} ({len(master_df)} trading days)")
    return master_df

if __name__ == "__main__":
    fetch_stock_data()

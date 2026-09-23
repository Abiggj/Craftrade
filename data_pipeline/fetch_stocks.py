"""
Automated Historical and Daily Stock Data Collector for CrafTrade.
Uses yfinance to fetch OHLCV and technical metrics for target Indian stocks and benchmark indices.
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Target Indian Stocks (NSE) and Benchmark Indices
TICKERS = {
    # IT Sector
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    # Banking Sector
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS",
    "IDBI": "IDBI.NS",
    # Benchmark Indices
    "NIFTY50": "^NSEI",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK"
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stocks")

def fetch_stock_data(start_date="2010-01-01", end_date=None):
    """
    Downloads historical OHLCV data for all tickers and saves individual CSVs
    and a consolidated dataframe.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"[*] Fetching stock data from {start_date} to {end_date}...")
    
    all_series = []
    
    for symbol, yahoo_ticker in TICKERS.items():
        print(f"    -> Downloading {symbol} ({yahoo_ticker})...")
        try:
            ticker_obj = yf.Ticker(yahoo_ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                print(f"       [!] Warning: No data returned for {yahoo_ticker}")
                continue
            
            # Format dataframe
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
            
            # Keep core columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            df.columns = ['Date', f'Open_{symbol}', f'High_{symbol}', f'Low_{symbol}', f'Close_{symbol}', f'Adj_Close_{symbol}', f'Volume_{symbol}']
            
            # Compute daily return % for quick impact calculation
            df[f'Pct_Change_{symbol}'] = ((df[f'Close_{symbol}'] - df[f'Open_{symbol}']) / df[f'Open_{symbol}']) * 100
            
            # Save individual stock file
            individual_file = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
            df.to_csv(individual_file, index=False)
            all_series.append(df)
            
        except Exception as e:
            print(f"       [!] Error fetching {yahoo_ticker}: {e}")

    if not all_series:
        print("[!] No data fetched.")
        return None

    # Merge all into one consolidated master stock dataset on Date
    print("[*] Merging all stocks into consolidated master dataset...")
    master_df = all_series[0]
    for df in all_series[1:]:
        master_df = pd.merge(master_df, df, on='Date', how='outer')
        
    master_df = master_df.sort_values(by='Date').reset_index(drop=True)
    master_file = os.path.join(OUTPUT_DIR, "master_stocks.csv")
    master_df.to_csv(master_file, index=False)
    
    print(f"[+] Successfully saved master stock data to {master_file} ({len(master_df)} trading days)")
    return master_df

if __name__ == "__main__":
    fetch_stock_data()

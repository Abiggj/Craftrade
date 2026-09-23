"""
Simulation Dataset Builder for CrafTrade.
Pairs historical news headlines with real next-day stock and sector price impacts.
Generates structured training datasets (CSV and JSONL) for local LLM simulation.
"""

import os
import ast
import json
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STOCKS_DIR = os.path.join(DATA_DIR, "stocks")
NEWS_DIR = os.path.join(DATA_DIR, "news")
OUTPUT_DIR = os.path.join(DATA_DIR, "dataset")

SECTOR_MAP = {
    "TCS": "IT Services",
    "INFY": "IT Services",
    "WIPRO": "IT Services",
    "HCLTECH": "IT Services",
    "TECHM": "IT Services",
    "HDFCBANK": "Banking & Finance",
    "ICICIBANK": "Banking & Finance",
    "SBIN": "Banking & Finance",
    "AXISBANK": "Banking & Finance",
    "IDBI": "Banking & Finance"
}

SECTOR_INDEX_MAP = {
    "IT Services": "NIFTY_IT",
    "Banking & Finance": "NIFTY_BANK"
}

PEERS_MAP = {
    "TCS": ["INFY", "WIPRO", "HCLTECH"],
    "INFY": ["TCS", "WIPRO", "TECHM"],
    "WIPRO": ["TCS", "INFY", "HCLTECH"],
    "HCLTECH": ["TCS", "INFY", "TECHM"],
    "TECHM": ["TCS", "INFY", "WIPRO"],
    "HDFCBANK": ["ICICIBANK", "SBIN", "AXISBANK"],
    "ICICIBANK": ["HDFCBANK", "SBIN", "AXISBANK"],
    "SBIN": ["HDFCBANK", "ICICIBANK", "AXISBANK"],
    "AXISBANK": ["HDFCBANK", "ICICIBANK", "SBIN"],
    "IDBI": ["SBIN", "AXISBANK", "ICICIBANK"]
}

def load_stock_matrix():
    """Loads master stock data or combines individual stock CSVs."""
    master_file = os.path.join(STOCKS_DIR, "master_stocks.csv")
    if os.path.exists(master_file):
        df = pd.read_csv(master_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values(by='Date').reset_index(drop=True)
    
    # Fallback to existing model_files/data.csv if available
    fallback_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_files", "data.csv")
    if os.path.exists(fallback_file):
        print(f"[*] Loading fallback stock data from {fallback_file}...")
        df = pd.read_csv(fallback_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values(by='Date').reset_index(drop=True)
        
    raise FileNotFoundError("Stock data not found! Please run fetch_stocks.py first.")

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stock_df = load_stock_matrix()
    trading_dates = stock_df['Date'].tolist()
    
    # Load parsed news
    news_file = os.path.join(NEWS_DIR, "historical_news_parsed.csv")
    if not os.path.exists(news_file):
        # Fallback to news.csv in root
        news_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "news.csv")
        if not os.path.exists(news_file):
            raise FileNotFoundError("Parsed news not found! Please run fetch_news.py first.")
            
    print(f"[*] Loading news from {news_file}...")
    news_df = pd.read_csv(news_file)
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    samples = []
    instruction_records = []
    
    print("[*] Correlating news events with next-day price impacts...")
    for idx, row in news_df.iterrows():
        news_date = row['Date']
        headline = row.get('News', row.get('news', ''))
        entities = row.get('Entities', [])
        if isinstance(entities, str):
            try:
                entities = ast.literal_eval(entities)
            except:
                entities = []
                
        # If no explicit entities detected, scan headline for common stock keywords
        if not entities:
            lower = str(headline).lower()
            for comp in SECTOR_MAP.keys():
                if comp.lower() in lower:
                    entities.append(comp)
                    
        if not entities:
            continue
            
        # Find next trading date
        future_dates = [d for d in trading_dates if d >= news_date]
        if len(future_dates) < 2:
            continue
            
        t_date = future_dates[0]
        t1_date = future_dates[1]
        
        row_t = stock_df[stock_df['Date'] == t_date].iloc[0]
        row_t1 = stock_df[stock_df['Date'] == t1_date].iloc[0]
        
        for stock in entities:
            close_col = f"Close_{stock}"
            open_col = f"Open_{stock}"
            high_col = f"High_{stock}"
            low_col = f"Low_{stock}"
            
            if close_col not in stock_df.columns or open_col not in stock_df.columns:
                continue
                
            close_t = row_t[close_col]
            open_t1 = row_t1[open_col]
            high_t1 = row_t1[high_col]
            low_t1 = row_t1[low_col]
            close_t1 = row_t1[close_col]
            
            if pd.isna(close_t) or pd.isna(close_t1) or close_t <= 0:
                continue
                
            gap_pct = ((open_t1 - close_t) / close_t) * 100
            day_return_pct = ((close_t1 - close_t) / close_t) * 100
            swing_pct = ((high_t1 - low_t1) / open_t1) * 100
            
            # Direction classification
            if day_return_pct > 1.2:
                direction = "Bullish / Surge"
            elif day_return_pct < -1.2:
                direction = "Bearish / Drop"
            elif swing_pct > 2.5:
                direction = "High Volatility / Uncertain"
            else:
                direction = "Neutral / Muted Reaction"
                
            sector = SECTOR_MAP.get(stock, "Equities")
            peers = PEERS_MAP.get(stock, [])
            
            sample = {
                "Date": news_date.strftime('%Y-%m-%d'),
                "Headline": str(headline)[:500],
                "Stock": stock,
                "Sector": sector,
                "Base_Price": round(float(close_t), 2),
                "Simulated_Next_Close": round(float(close_t1), 2),
                "Gap_Pct": round(float(gap_pct), 2),
                "Return_Pct": round(float(day_return_pct), 2),
                "Intraday_Swing_Pct": round(float(swing_pct), 2),
                "Direction": direction
            }
            samples.append(sample)
            
            # Formulate structured LLM simulation pair
            inst_obj = {
                "instruction": "You are a quantitative market simulation analyst. Given the financial news headline, identify the affected stock, simulate the next-day price impact, directional trend, and explain the underlying market mechanism so the user learns how news drives stock movements.",
                "input": f"News Headline: \"{str(headline)[:300]}\"",
                "output": {
                    "primary_stock_affected": stock,
                    "sector": sector,
                    "projected_movement": direction,
                    "estimated_gap_percent": round(float(gap_pct), 2),
                    "estimated_day_return_percent": round(float(day_return_pct), 2),
                    "expected_volatility": "High" if swing_pct > 2.0 else "Normal",
                    "sector_peer_spillover": peers[:3],
                    "educational_explanation": (
                        f"When news like this breaks regarding {stock}, markets typically react with a {direction.lower()} bias. "
                        f"On the next trading session, the stock experienced an estimated return of {round(float(day_return_pct), 2)}% "
                        f"with an intraday price swing of {round(float(swing_pct), 2)}%. Peers in the {sector} sector ({', '.join(peers[:3])}) "
                        f"often experience secondary sympathy movements due to shared sectoral sentiment."
                    )
                }
            }
            instruction_records.append(inst_obj)

    # Save to CSV
    csv_out = os.path.join(OUTPUT_DIR, "simulation_dataset.csv")
    df_samples = pd.DataFrame(samples)
    df_samples.to_csv(csv_out, index=False)
    
    # Save to JSONL
    jsonl_out = os.path.join(OUTPUT_DIR, "simulation_instruction_dataset.jsonl")
    with open(jsonl_out, 'w', encoding='utf-8') as f:
        for item in instruction_records:
            f.write(json.dumps(item) + "\n")
            
    print(f"[+] Dataset created successfully!")
    print(f"    - Tabular dataset: {csv_out} ({len(df_samples)} examples)")
    print(f"    - LLM instruction dataset: {jsonl_out} ({len(instruction_records)} training records)")

if __name__ == "__main__":
    build_dataset()

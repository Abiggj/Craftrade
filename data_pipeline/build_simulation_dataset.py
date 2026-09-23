"""
Simulation Dataset Builder for CrafTrade.
Pairs high-volume multi-source news headlines with real next-day market responses:
- Corporate level: TCS, INFY, HDFC Bank, etc.
- Systemic / Macro level: NIFTY 50, SENSEX, INDIA VIX
Generates clean tabular dataset (CSV) and LLM instruction dataset (JSONL).
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
    "KOTAKBANK": "Banking & Finance",
    "RELIANCE": "Energy & Conglomerate",
    "TATAMOTORS": "Automobiles",
    "NIFTY50": "Macro / Benchmark",
    "SENSEX": "Macro / Benchmark",
    "SYSTEMIC": "Sovereign / Macro"
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
    "RELIANCE": ["TCS", "HDFCBANK", "INFY"],
    "SYSTEMIC": ["NIFTY50", "SENSEX", "NIFTY_BANK"]
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
    
    # Priority order for finding news files
    candidates = [
        os.path.join(NEWS_DIR, "master_financial_news.csv"),
        os.path.join(NEWS_DIR, "historical_news_parsed.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "news.csv")
    ]
    
    news_file = None
    for cand in candidates:
        if os.path.exists(cand):
            news_file = cand
            break
            
    if not news_file:
        raise FileNotFoundError("Parsed news not found! Please run fetch_news.py first.")
            
    print(f"[*] Ingesting news from {news_file}...")
    news_df = pd.read_csv(news_file)
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')
    news_df = news_df.dropna(subset=['Date', 'News'])
    
    samples = []
    instruction_records = []
    
    print(f"[*] Correlating {len(news_df):,} news events with real market responses...")
    for idx, row in news_df.iterrows():
        news_date = row['Date']
        headline = str(row.get('News', row.get('news', ''))).strip()
        entities = row.get('Entities', [])
        
        if isinstance(entities, str):
            try:
                entities = ast.literal_eval(entities)
            except:
                entities = []
                
        if not entities:
            lower = headline.lower()
            for comp in SECTOR_MAP.keys():
                if comp.lower() in lower:
                    entities.append(comp)
                    
        if not entities:
            continue
            
        future_dates = [d for d in trading_dates if d >= news_date]
        if len(future_dates) < 2:
            continue
            
        t_date = future_dates[0]
        t1_date = future_dates[1]
        
        row_t = stock_df[stock_df['Date'] == t_date].iloc[0]
        row_t1 = stock_df[stock_df['Date'] == t1_date].iloc[0]
        
        for stock in entities:
            eval_ticker = "NIFTY50" if stock == "SYSTEMIC" else stock
            close_col = f"Close_{eval_ticker}"
            open_col = f"Open_{eval_ticker}"
            high_col = f"High_{eval_ticker}"
            low_col = f"Low_{eval_ticker}"
            
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
                "Headline": headline[:500],
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
            
            inst_obj = {
                "instruction": "You are an institutional quantitative market simulation strategist. Given the financial or political event, analyze the impact on benchmark indices and target stocks.",
                "input": f"Event Headline: \"{headline[:300]}\"",
                "output": {
                    "target_entity": stock,
                    "sector_scope": sector,
                    "projected_movement": direction,
                    "estimated_gap_percent": round(float(gap_pct), 2),
                    "estimated_day_return_percent": round(float(day_return_pct), 2),
                    "intraday_volatility_swing": round(float(swing_pct), 2),
                    "correlated_assets": peers[:3]
                }
            }
            instruction_records.append(inst_obj)

    # Save outputs
    csv_out = os.path.join(OUTPUT_DIR, "simulation_dataset.csv")
    df_samples = pd.DataFrame(samples)
    df_samples.to_csv(csv_out, index=False)
    
    jsonl_out = os.path.join(OUTPUT_DIR, "simulation_instruction_dataset.jsonl")
    with open(jsonl_out, 'w', encoding='utf-8') as f:
        for item in instruction_records:
            f.write(json.dumps(item) + "\n")
            
    print("=" * 68)
    print(f"[+] SIMULATION TRAINING DATASET COMPILED!")
    print(f"    Total Correlated Event Pairs: {len(df_samples):,}")
    print(f"    Tabular CSV Output: {csv_out}")
    print(f"    Instruction JSONL Output: {jsonl_out}")
    print("=" * 68)

if __name__ == "__main__":
    build_dataset()

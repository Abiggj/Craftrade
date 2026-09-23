"""
High-Volume Multi-Source Financial News Aggregator for CrafTrade.
Dramatically expands news volume and diversity across:
1. Multi-Publisher Financial RSS Streams (Economic Times, LiveMint, MoneyControl, Financial Express)
2. Topic-Specific Google News Aggregation (Macro, Political, Banking, IT, Regulatory, Global Shocks)
3. Direct Yahoo Finance Corporate News Feeds for All Tickers
4. Comprehensive Historical Shock Event Registry (1998 - 2024)
5. Parsing and Segmenting Existing Historical News Archives
"""

import os
import re
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NEWS_OUTPUT_DIR = os.path.join(DATA_DIR, "news")
RAW_NEWS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "news")

COMPANY_KEYWORD_MAP = {
    "TCS": ["tcs", "tata consultancy", "rajesh gopinathan", "krithivasan"],
    "INFY": ["infosys", "infy", "salil parekh", "narayana murthy", "nandan nilekani", "vishal sikka"],
    "WIPRO": ["wipro", "thierry delaporte", "rishad premji", "azim premji", "srini pallia"],
    "HCLTECH": ["hcl tech", "hcl", "roshni nadar", "c vijayakumar"],
    "TECHM": ["tech mahindra", "techm", "cp gurnani", "mohit joshi"],
    "HDFCBANK": ["hdfc bank", "hdfc", "sashidhar jagdishan", "aditya puri"],
    "ICICIBANK": ["icici bank", "icici", "sandeep bakhshi", "chanda kochhar"],
    "SBIN": ["sbi", "state bank of india", "dinesh khara", "cs setty"],
    "AXISBANK": ["axis bank", "amitabh chaudhry"],
    "KOTAKBANK": ["kotak mahindra", "kotak bank", "uday kotak", "ashok vaswani"],
    "RELIANCE": ["reliance industries", "mukesh ambani", "jio financial"],
    "TATAMOTORS": ["tata motors", "jlr", "jaguar land rover"],
    "SYSTEMIC": ["modi", "prime minister", "parliament", "election", "rbi", "repo rate", "budget", "sebi", "war"]
}

# Multi-Source Financial RSS Feeds
EXPANDED_RSS_FEEDS = [
    # Economic Times
    ("Economic Times - Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Economic Times - Stocks", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("Economic Times - Economy", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    
    # LiveMint
    ("LiveMint - Markets", "https://www.livemint.com/rss/markets"),
    ("LiveMint - Companies", "https://www.livemint.com/rss/companies"),
    
    # MoneyControl
    ("MoneyControl - Top News", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("MoneyControl - Business", "https://www.moneycontrol.com/rss/business.xml"),
    ("MoneyControl - Market Reports", "https://www.moneycontrol.com/rss/marketreports.xml"),
    
    # Financial Express
    ("Financial Express - Market", "https://www.financialexpress.com/market/feed/")
]

# Targeted Google News Query Feeds
TARGETED_GOOGLE_QUERIES = [
    "Nifty+50+Sensex+crash+OR+surge+stock+market",
    "RBI+repo+rate+inflation+policy+announcement",
    "Prime+Minister+Cabinet+India+economic+policy",
    "TCS+OR+Infosys+OR+Wipro+quarterly+earnings+CEO",
    "HDFC+Bank+OR+ICICI+Bank+OR+SBI+banking+credit",
    "FII+foreign+investors+selling+buying+India+equities",
    "SEBI+regulatory+investigation+crackdown+market"
]

# Curated Historical Market Shocks & Systemic Turning Points (1998 - 2024)
CURATED_HISTORICAL_SHOCKS = [
    {"Date": "2000-03-01", "News": "Dot-Com bubble burst triggers global tech rout; Indian IT software stocks face severe valuation contraction", "Entities": ["TCS", "INFY", "WIPRO", "SYSTEMIC"]},
    {"Date": "2001-03-02", "News": "Ketan Parekh stock market scam unfolds; SEBI initiates investigation as markets plunge into panic", "Entities": ["SYSTEMIC"]},
    {"Date": "2004-05-17", "News": "Surprise NDA election defeat and political uncertainty triggers 15% market crash; trading halted on lower circuits", "Entities": ["SYSTEMIC", "SBIN"]},
    {"Date": "2008-01-22", "News": "Subprime global liquidity contagion triggers massive selloff; Nifty drops 10% in extreme intraday volatility", "Entities": ["SYSTEMIC", "HDFCBANK", "ICICIBANK"]},
    {"Date": "2008-09-15", "News": "Lehman Brothers files for Chapter 11 bankruptcy; global financial crisis sparks panic flight from emerging markets", "Entities": ["SYSTEMIC", "ICICIBANK"]},
    {"Date": "2009-01-07", "News": "Satyam founder Ramalinga Raju confesses to massive Rs 7,000 crore accounting fraud; corporate governance crisis", "Entities": ["WIPRO", "INFY", "TCS", "SYSTEMIC"]},
    {"Date": "2009-05-18", "News": "UPA wins decisive parliamentary majority; Indian equity markets trigger upper circuit halt within 30 seconds of open", "Entities": ["SYSTEMIC", "SBIN", "RELIANCE"]},
    {"Date": "2013-08-28", "News": "US Federal Reserve Taper Tantrum drives Indian Rupee to record low against US dollar; acute foreign capital flight", "Entities": ["SYSTEMIC", "HDFCBANK"]},
    {"Date": "2014-05-16", "News": "Modi-led BJP wins historic single-party majority; Sensex and Nifty surge to record highs on economic reform optimism", "Entities": ["SYSTEMIC", "SBIN", "RELIANCE"]},
    {"Date": "2015-08-24", "News": "Black Monday: China Yuan devaluation and global growth fears trigger 1,600 point intraday crash in Sensex", "Entities": ["SYSTEMIC"]},
    {"Date": "2016-11-08", "News": "Prime Minister Modi announces sudden demonetization of 500 and 1000 rupee notes; cash velocity shock hits real estate and banking", "Entities": ["SYSTEMIC", "SBIN", "HDFCBANK"]},
    {"Date": "2017-08-18", "News": "Infosys CEO Vishal Sikka resigns unexpectedly amid governance dispute with founders; stock crashes 9.6%", "Entities": ["INFY", "TCS", "WIPRO"]},
    {"Date": "2018-09-21", "News": "IL&FS defaults on debt obligations sparking severe shadow banking credit crisis; housing finance and NBFCs plunge", "Entities": ["SYSTEMIC", "HDFCBANK", "AXISBANK"]},
    {"Date": "2019-02-14", "News": "Pulwama geopolitical escalation sparks cross-border military tensions; heightened risk premium across equities", "Entities": ["SYSTEMIC"]},
    {"Date": "2019-09-20", "News": "Finance Minister announces surprise massive corporate tax cut from 30% to 22%; market surges 5.3% in biggest gain in a decade", "Entities": ["SYSTEMIC", "RELIANCE", "SBIN", "HDFCBANK"]},
    {"Date": "2020-03-12", "News": "WHO declares COVID-19 pandemic; global circuit breakers triggered as equity markets enter bear market territory", "Entities": ["SYSTEMIC", "TCS", "HDFCBANK", "INFY"]},
    {"Date": "2020-03-23", "News": "Prime Minister announces nationwide lockdown; Nifty crashes 13% hitting lower circuit as trading is suspended", "Entities": ["SYSTEMIC", "SBIN", "ICICIBANK", "RELIANCE"]},
    {"Date": "2022-02-24", "News": "Russia launches full-scale military invasion of Ukraine; Brent crude spikes above $105 per barrel driving broad risk-off", "Entities": ["SYSTEMIC", "RELIANCE"]},
    {"Date": "2022-05-04", "News": "RBI Governor announces surprise off-cycle 40 bps repo rate hike to combat runaway inflation; banks and rate-sensitives slide", "Entities": ["SYSTEMIC", "HDFCBANK", "SBIN"]},
    {"Date": "2023-01-25", "News": "Hindenburg Research releases scathing report on Adani Group alleging accounting fraud; conglomerate shares tumble", "Entities": ["SYSTEMIC", "SBIN"]},
    {"Date": "2023-03-16", "News": "TCS CEO Rajesh Gopinathan resigns unexpectedly; K Krithivasan named CEO designate to ensure transition stability", "Entities": ["TCS", "INFY", "WIPRO"]},
    {"Date": "2024-06-04", "News": "General Election vote count reveals ruling coalition falls short of 400-seat projection; Nifty crashes 1,380 points in historic volatility", "Entities": ["SYSTEMIC", "SBIN", "RELIANCE", "HDFCBANK"]}
]

def clean_text(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)  # remove html
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_entities(headline):
    lower = headline.lower()
    detected = []
    for company, keywords in COMPANY_KEYWORD_MAP.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                detected.append(company)
                break
    return list(set(detected))

def fetch_all_rss_streams():
    """Fetches articles across all configured Indian financial RSS feeds."""
    headers = {"User-Agent": "Mozilla/5.0 (CrafTrade News Aggregator v2.0)"}
    records = []
    print("[*] 1/4 Ingesting Multi-Publisher Financial RSS Streams...")

    for label, feed_url in EXPANDED_RSS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=headers, timeout=12)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                count = 0
                for item in root.findall('.//item'):
                    title_elem = item.find('title')
                    date_elem = item.find('pubDate')
                    if title_elem is not None and title_elem.text:
                        hl = clean_text(title_elem.text)
                        dt = date_elem.text if date_elem is not None else datetime.today().strftime('%Y-%m-%d')
                        entities = detect_entities(hl)
                        records.append({
                            "Date": dt,
                            "News": hl,
                            "Entities": entities,
                            "Source": label
                        })
                        count += 1
                print(f"    -> {label}: {count} headlines retrieved.")
        except Exception as e:
            print(f"    [!] Error pulling {label}: {e}")

    return records

def fetch_targeted_google_news():
    """Pulls targeted topic feeds from Google News."""
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []
    print("[*] 2/4 Ingesting Targeted Macro, Political, and Sector Feeds...")

    for query in TARGETED_GOOGLE_QUERIES:
        url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                count = 0
                for item in root.findall('.//item'):
                    title_elem = item.find('title')
                    date_elem = item.find('pubDate')
                    if title_elem is not None and title_elem.text:
                        hl = clean_text(title_elem.text)
                        dt = date_elem.text if date_elem is not None else datetime.today().strftime('%Y-%m-%d')
                        entities = detect_entities(hl)
                        records.append({
                            "Date": dt,
                            "News": hl,
                            "Entities": entities,
                            "Source": f"GoogleNews: {query[:25]}"
                        })
                        count += 1
                print(f"    -> Query [{query[:30]}...]: {count} headlines.")
        except Exception as e:
            print(f"    [!] Error pulling query {query}: {e}")

    return records

def fetch_yfinance_ticker_news():
    """Pulls institutional company news directly via Yahoo Finance."""
    if yf is None:
        return []
    
    records = []
    print("[*] 3/4 Ingesting Ticker-Specific Feeds via Yahoo Finance...")
    tickers = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "RELIANCE.NS"]
    
    for t_sym in tickers:
        try:
            t = yf.Ticker(t_sym)
            news_items = getattr(t, "news", [])
            for item in news_items:
                title = item.get("title", "")
                if title:
                    dt = datetime.fromtimestamp(item.get("providerPublishTime", datetime.today().timestamp())).strftime('%Y-%m-%d')
                    clean_hl = clean_text(title)
                    records.append({
                        "Date": dt,
                        "News": clean_hl,
                        "Entities": [t_sym.replace(".NS", "")],
                        "Source": f"YahooFinance ({t_sym})"
                    })
            print(f"    -> {t_sym}: {len(news_items)} corporate news items.")
        except Exception as e:
            print(f"    [!] Error pulling yfinance news for {t_sym}: {e}")

    return records

def parse_existing_archives():
    """Segments existing historical archives in ./news/ into discrete records."""
    print("[*] 4/4 Parsing Existing News Text Archives...")
    os.makedirs(NEWS_OUTPUT_DIR, exist_ok=True)
    records = []

    archive_files = glob.glob(os.path.join(RAW_NEWS_DIR, "*.txt"))
    for filepath in archive_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            current_date = None
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                date_match = re.match(r'^(\d{1,2})-(\d{1,2})-(\d{4})$', line_str)
                if date_match:
                    d, m, y = date_match.groups()
                    current_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                    continue
                
                if current_date and len(line_str.split()) >= 4:
                    records.append({
                        "Date": current_date,
                        "News": line_str[:1500],
                        "Entities": detect_entities(line_str),
                        "Source": "HistoricalArchive"
                    })
        except Exception as e:
            print(f"    [!] Error parsing archive {filepath}: {e}")

    return records

def run_pipeline():
    os.makedirs(NEWS_OUTPUT_DIR, exist_ok=True)
    
    all_data = []
    # 1. Curated Shocks
    for shock in CURATED_HISTORICAL_SHOCKS:
        all_data.append({
            "Date": shock["Date"],
            "News": shock["News"],
            "Entities": shock["Entities"],
            "Source": "CuratedHistoricalShocks"
        })
    print(f"[+] Loaded {len(CURATED_HISTORICAL_SHOCKS)} curated historical turning point events.")

    # 2. RSS
    all_data.extend(fetch_all_rss_streams())
    
    # 3. Google News
    all_data.extend(fetch_targeted_google_news())
    
    # 4. YFinance
    all_data.extend(fetch_yfinance_ticker_news())
    
    # 5. Existing Archives
    all_data.extend(parse_existing_archives())

    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=["News"])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)

        master_news_file = os.path.join(NEWS_OUTPUT_DIR, "master_financial_news.csv")
        df.to_csv(master_news_file, index=False)
        print("=" * 68)
        print(f"[+] HIGH-VOLUME NEWS AGGREGATION COMPLETE!")
        print(f"    Total Distinct Financial News Records: {len(df):,}")
        print(f"    Destination: {master_news_file}")
        print("=" * 68)
        return df

if __name__ == "__main__":
    run_pipeline()

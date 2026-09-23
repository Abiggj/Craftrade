"""
High-Volume Multi-Source Financial News Aggregator for CrafTrade.
Extracts hundreds of thousands of historical headlines till date:
1. High-Precision Parser for the 100+ local archives in ./news/ (2010 - 2024)
   - Splits concatenated daily headline streams into discrete, deduplicated headlines.
2. Live & Recent Financial News Aggregator (Till Date) via Multi-Source RSS & Google News
3. Direct Yahoo Finance Corporate News Feeds
4. Curated Historical Turning Points & Black Swan Shocks (1998 - 2024)
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
    "KOTAKBANK": ["kotak mahindra", "kotak bank", "uday kotak"],
    "RELIANCE": ["reliance industries", "mukesh ambani", "jio financial"],
    "TATAMOTORS": ["tata motors", "jlr"],
    "SYSTEMIC": ["modi", "prime minister", "parliament", "election", "rbi", "repo rate", "budget", "sebi", "war", "nifty", "sensex", "crash", "surge", "lockdown"]
}

EXPANDED_RSS_FEEDS = [
    ("Economic Times - Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Economic Times - Stocks", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("Economic Times - Economy", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    ("LiveMint - Markets", "https://www.livemint.com/rss/markets"),
    ("LiveMint - Companies", "https://www.livemint.com/rss/companies"),
    ("MoneyControl - Top News", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("MoneyControl - Business", "https://www.moneycontrol.com/rss/business.xml"),
    ("Financial Express - Market", "https://www.financialexpress.com/market/feed/")
]

TARGETED_GOOGLE_QUERIES = [
    "Nifty+50+Sensex+crash+OR+surge+stock+market",
    "RBI+repo+rate+inflation+policy+announcement",
    "Prime+Minister+Cabinet+India+economic+policy",
    "TCS+OR+Infosys+OR+Wipro+quarterly+earnings+CEO",
    "HDFC+Bank+OR+ICICI+Bank+OR+SBI+banking+credit",
    "FII+foreign+investors+selling+buying+India+equities",
    "SEBI+regulatory+investigation+crackdown+market"
]

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
    text = re.sub(r'<[^>]+>', ' ', text)
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

def split_daily_stream_into_headlines(stream_text: str):
    """
    Splits the long concatenated daily text dump in news/*.txt into discrete headlines.
    Handles duplicate phrases and headline boundaries.
    """
    stream_text = re.sub(r'\s+', ' ', stream_text).strip()
    # Split on boundary: lowercase or punctuation followed by capital word
    chunks = re.split(r'(?<=[a-z0-9\.\?\!\'\”\’\)])\s+(?=[A-Z][a-z])', stream_text)
    
    clean_list = []
    seen = set()
    for item in chunks:
        item = item.strip()
        if len(item) < 18 or len(item.split()) < 4:
            continue
            
        # Deduplicate duplicated strings within chunk (e.g. "Headline Headline")
        words = item.split()
        half = len(words) // 2
        if half >= 4 and " ".join(words[:half]) == " ".join(words[half:2*half]):
            item = " ".join(words[:half])
            
        lower_item = item.lower()
        if lower_item not in seen:
            seen.add(lower_item)
            clean_list.append(item)
            
    return clean_list

def parse_local_historical_archives():
    """
    Parses all ~100 archive files in ./news/ (spanning 2010 to 2024).
    Extracts hundreds of thousands of discrete headlines.
    """
    archive_files = glob.glob(os.path.join(RAW_NEWS_DIR, "*.txt"))
    print(f"[*] 1/4 Parsing {len(archive_files)} Local Archive Files in ./news/ (2010 - 2024)...")
    
    records = []
    total_parsed = 0
    
    for filepath in archive_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            current_date = None
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                # Check for date line: D-M-YYYY or DD-MM-YYYY
                date_match = re.match(r'^(\d{1,2})-(\d{1,2})-(\d{4})$', line_str)
                if date_match:
                    d, m, y = date_match.groups()
                    current_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                    continue
                
                if current_date:
                    headlines = split_daily_stream_into_headlines(line_str)
                    for hl in headlines:
                        records.append({
                            "Date": current_date,
                            "News": hl,
                            "Entities": detect_entities(hl),
                            "Source": "ET_Local_Archive"
                        })
                    total_parsed += len(headlines)
        except Exception as e:
            print(f"    [!] Error parsing {filepath}: {e}")
            
    print(f"[+] Successfully extracted {total_parsed:,} individual headlines from local archives!")
    return records

def fetch_rss_and_web_feeds():
    """Fetches real-time feeds till date."""
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []
    print("[*] 2/4 Ingesting Multi-Source Financial RSS Streams (Till Date)...")

    for label, feed_url in EXPANDED_RSS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                count = 0
                for item in root.findall('.//item'):
                    title = item.find('title')
                    pdate = item.find('pubDate')
                    if title is not None and title.text:
                        hl = clean_text(title.text)
                        dt = pdate.text if pdate is not None else datetime.today().strftime('%Y-%m-%d')
                        records.append({
                            "Date": dt,
                            "News": hl,
                            "Entities": detect_entities(hl),
                            "Source": label
                        })
                        count += 1
                print(f"    -> {label}: {count} live headlines.")
        except Exception:
            continue

    print("[*] 3/4 Ingesting Google News Multi-Topic Streams...")
    for query in TARGETED_GOOGLE_QUERIES:
        url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item'):
                    title = item.find('title')
                    pdate = item.find('pubDate')
                    if title is not None and title.text:
                        hl = re.sub(r' - [^-]+$', '', clean_text(title.text))
                        records.append({
                            "Date": pdate.text if pdate is not None else datetime.today().strftime('%Y-%m-%d'),
                            "News": hl,
                            "Entities": detect_entities(hl),
                            "Source": "GoogleNews"
                        })
        except Exception:
            continue

    return records

def fetch_yfinance_news():
    """Pulls recent corporate press items via yfinance."""
    if yf is None:
        return []
    records = []
    print("[*] 4/4 Ingesting YFinance Corporate News...")
    for sym in ["TCS.NS", "INFY.NS", "WIPRO.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "RELIANCE.NS"]:
        try:
            t = yf.Ticker(sym)
            for item in getattr(t, "news", []):
                title = item.get("title", "")
                if title:
                    dt = datetime.fromtimestamp(item.get("providerPublishTime", datetime.today().timestamp())).strftime('%Y-%m-%d')
                    records.append({
                        "Date": dt,
                        "News": clean_text(title),
                        "Entities": [sym.replace(".NS", "")],
                        "Source": f"YFinance_{sym}"
                    })
        except Exception:
            continue
    return records

def run_pipeline():
    os.makedirs(NEWS_OUTPUT_DIR, exist_ok=True)
    all_news = []

    # 1. Curated Shocks
    for shock in CURATED_HISTORICAL_SHOCKS:
        all_news.append({
            "Date": shock["Date"],
            "News": shock["News"],
            "Entities": shock["Entities"],
            "Source": "CuratedHistoricalShocks"
        })

    # 2. Local archives (2010 - 2024) -> Hundreds of thousands of headlines!
    all_news.extend(parse_local_historical_archives())

    # 3. Live RSS & Google News
    all_news.extend(fetch_rss_and_web_feeds())

    # 4. YFinance
    all_news.extend(fetch_yfinance_news())

    # Check for optional Kaggle dataset if user downloaded it
    kaggle_csv = os.path.join(NEWS_OUTPUT_DIR, "india-news-headlines.csv")
    if os.path.exists(kaggle_csv):
        print(f"[*] Found Kaggle master news file {kaggle_csv}. Ingesting...")
        try:
            k_df = pd.read_csv(kaggle_csv)
            # Standard columns: publish_date (YYYYMMDD), headline_category, headline_text
            if 'headline_text' in k_df.columns and 'publish_date' in k_df.columns:
                # Filter business/market categories or keywords to keep it focused
                k_df = k_df.dropna(subset=['headline_text'])
                k_records = []
                for _, r in k_df.iterrows():
                    h_text = str(r['headline_text'])
                    ents = detect_entities(h_text)
                    if ents:
                        dt_str = str(r['publish_date'])
                        if len(dt_str) == 8:
                            fmt_date = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
                            k_records.append({
                                "Date": fmt_date,
                                "News": h_text,
                                "Entities": ents,
                                "Source": "Kaggle_TOI_Archive"
                            })
                all_news.extend(k_records)
                print(f"[+] Added {len(k_records):,} corporate/market headlines from Kaggle dataset.")
        except Exception as e:
            print(f"[!] Error reading Kaggle CSV: {e}")

    df = pd.DataFrame(all_news)
    if not df.empty:
        df = df.drop_duplicates(subset=["News"])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'News']).sort_values(by="Date", ascending=False).reset_index(drop=True)

        master_file = os.path.join(NEWS_OUTPUT_DIR, "master_financial_news.csv")
        df.to_csv(master_file, index=False)
        print("=" * 74)
        print(f"[+] MULTI-YEAR NEWS AGGREGATION COMPLETE!")
        print(f"    Total Consolidated Headlines: {len(df):,}")
        print(f"    Date Range Covered: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"    Saved to: {master_file}")
        print("=" * 74)
        return df

if __name__ == "__main__":
    run_pipeline()

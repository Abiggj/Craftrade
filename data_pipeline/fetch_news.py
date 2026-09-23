"""
Automated Financial News Ingestion and Historical Archive Parser for CrafTrade.
- Scrapes live/recent RSS feeds for Indian financial news (no browser/chromedriver needed).
- Parses and extracts structured headlines with dates from historical news text archives.
"""

import os
import re
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NEWS_OUTPUT_DIR = os.path.join(DATA_DIR, "news")
RAW_NEWS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "news")

# Keywords mapped to target companies and sectors
COMPANY_KEYWORD_MAP = {
    "TCS": ["tcs", "tata consultancy", "rajesh gopinathan", "krithivasan"],
    "INFY": ["infosys", "infy", "salil parekh", "narayana murthy", "nandan nilekani", "vishal sikka"],
    "WIPRO": ["wipro", "thierry delaporte", "rishad premji", "azim premji"],
    "HCLTECH": ["hcl tech", "hcl", "roshni nadar", "c vijayakumar"],
    "TECHM": ["tech mahindra", "techm", "cp gurnani", "mohit joshi"],
    "HDFCBANK": ["hdfc bank", "hdfc", "sashidhar jagdishan", "aditya puri"],
    "ICICIBANK": ["icici bank", "icici", "sandeep bakhshi", "chanda kochhar"],
    "SBIN": ["sbi", "state bank of india", "dinesh khara"],
    "AXISBANK": ["axis bank", "amitabh chaudhry"],
    "IDBI": ["idbi bank", "idbi"]
}

# RSS feeds for ongoing news
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://news.google.com/rss/search?q=TCS+stock+OR+Infosys+stock+OR+Nifty+when:2d&hl=en-IN&gl=IN&ceid=IN:en"
]

def clean_headline(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_entities(headline):
    """Detects which target stocks are mentioned in the headline."""
    lower_hl = headline.lower()
    detected = []
    for company, keywords in COMPANY_KEYWORD_MAP.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', lower_hl):
                detected.append(company)
                break
    return list(set(detected))

def parse_historical_news_archives():
    """
    Parses all txt files in the existing ./news/ directory.
    Extracts (date, headline) pairs and detects mentioned companies.
    """
    os.makedirs(NEWS_OUTPUT_DIR, exist_ok=True)
    all_records = []
    
    archive_files = glob.glob(os.path.join(RAW_NEWS_DIR, "*.txt"))
    print(f"[*] Found {len(archive_files)} historical news archive files. Parsing...")
    
    for filepath in archive_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            current_date = None
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                # Check for date line (e.g. 1-3-2024 or 01-03-2024 or 2024-03-01)
                date_match = re.match(r'^(\d{1,2})-(\d{1,2})-(\d{4})$', line_str)
                if date_match:
                    d, m, y = date_match.groups()
                    current_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                    continue
                
                if current_date:
                    # In these files, headlines are often concatenated in long strings.
                    # We segment long streams into coherent headlines.
                    # Split on typical sentence boundaries or word-length chunks
                    content = line_str
                    # Extract individual news phrases (minimum 5 words)
                    words = content.split()
                    if len(words) >= 5:
                        entities = detect_entities(content)
                        all_records.append({
                            "Date": current_date,
                            "News": content[:2000],  # cap length per entry
                            "Entities": entities
                        })
        except Exception as e:
            print(f"    [!] Error parsing {filepath}: {e}")

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.drop_duplicates(subset=["Date", "News"])
        output_csv = os.path.join(NEWS_OUTPUT_DIR, "historical_news_parsed.csv")
        df.to_csv(output_csv, index=False)
        print(f"[+] Saved {len(df)} historical news records to {output_csv}")
    return df

def fetch_live_rss_news():
    """Fetches real-time financial news from RSS feeds."""
    live_records = []
    headers = {"User-Agent": "Mozilla/5.0 (CrafTrade News Aggregator)"}
    print("[*] Fetching live financial news from RSS feeds...")

    for feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item'):
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    title_text = title.text if title is not None else ""
                    title_text = clean_headline(title_text)
                    
                    if title_text:
                        entities = detect_entities(title_text)
                        live_records.append({
                            "Published": pub_date.text if pub_date is not None else "",
                            "Headline": title_text,
                            "Entities": entities
                        })
        except Exception as e:
            print(f"    [!] RSS Fetch error ({feed_url}): {e}")

    df_live = pd.DataFrame(live_records)
    live_file = os.path.join(NEWS_OUTPUT_DIR, "live_news.csv")
    df_live.to_csv(live_file, index=False)
    print(f"[+] Fetched {len(df_live)} live headlines saved to {live_file}")
    return df_live

if __name__ == "__main__":
    parse_historical_news_archives()
    fetch_live_rss_news()

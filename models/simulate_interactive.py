"""
Interactive Market Impact Simulation Engine for CrafTrade.
Simulates market impact for custom news headlines (e.g., 'today tcs ceo resigned').
Provides:
1. Fast local quantitative forecast (Price % return, Volatility, Direction)
2. Sector and peer stock spillover (INFY, WIPRO, Nifty IT, etc.)
3. Educational reasoning (via Local LLM with Ollama or built-in financial logic)
"""

import os
import sys
import json
import pickle
import requests
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_files", "local_simulator.pkl")

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

HISTORICAL_PRECEDENTS = {
    "ceo_resignation": (
        "Sudden executive departures (like Vishal Sikka from Infosys in 2017 or Rajesh Gopinathan from TCS in 2023) "
        "introduce governance and execution uncertainty. Markets typically penalize the stock 1.5% to 5% intraday "
        "until an interim successor and corporate guidance are formally confirmed."
    ),
    "earnings_beat": (
        "Strong quarterly beats, particularly in operating margins and constant-currency revenue growth, "
        "trigger institutional buying and short-covering, leading to 2% to 6% positive re-ratings."
    ),
    "regulatory_penalty": (
        "Regulatory sanctions (RBI compliance directives, penalties, or auditing scrutiny) cause sharp risk-off "
        "reactions in financials, triggering 2% to 8% drawdowns and increased hedging volume."
    )
}

class MarketSimulator:
    def __init__(self, model_file=MODEL_PATH):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found! Run train_local_simulator.py first.")
        with open(model_file, 'rb') as f:
            self.artifacts = pickle.load(f)
            
        self.stock_clf = self.artifacts["stock_classifier"]
        self.dir_clf = self.artifacts["direction_classifier"]
        self.ret_reg = self.artifacts["return_regressor"]
        self.swing_reg = self.artifacts["swing_regressor"]

    def simulate_event(self, headline: str):
        """Runs fast quantitative simulation on the input news text."""
        # 1. Primary entity & direction
        stock = self.stock_clf.predict([headline])[0]
        direction = self.dir_clf.predict([headline])[0]
        
        # Keyword override if user explicitly named a company
        lower_hl = headline.lower()
        for comp in SECTOR_MAP.keys():
            if comp.lower() in lower_hl:
                stock = comp
                break

        # 2. Predicted Price Metrics
        predicted_return = float(self.ret_reg.predict([headline])[0])
        predicted_swing = float(self.swing_reg.predict([headline])[0])
        
        # Apply directional heuristics if headline is clearly negative (e.g. resigned, raided, penalty)
        negative_words = ["resigned", "quit", "fired", "penalty", "fraud", "loss", "plunge", "fall", "scam"]
        positive_words = ["profit", "beat", "deal", "surges", "growth", "jump", "record", "dividend"]
        
        if any(w in lower_hl for w in negative_words) and predicted_return > 0:
            predicted_return = -abs(predicted_return)
            direction = "Bearish / Drop"
        elif any(w in lower_hl for w in positive_words) and predicted_return < 0:
            predicted_return = abs(predicted_return)
            direction = "Bullish / Surge"

        sector = SECTOR_MAP.get(stock, "Equities")
        peers = PEERS_MAP.get(stock, [])
        
        # Correlated peer impact estimate (beta dampening)
        peer_impacts = {
            peer: round(predicted_return * 0.45, 2) for peer in peers[:3]
        }

        result = {
            "headline": headline,
            "primary_stock": stock,
            "sector": sector,
            "projected_direction": direction,
            "estimated_price_change_pct": round(predicted_return, 2),
            "expected_intraday_swing_pct": round(max(predicted_swing, 1.2), 2),
            "peer_sector_spillover": peer_impacts
        }
        return result

    def get_educational_explanation(self, sim_result: dict, use_ollama: bool = True, ollama_model: str = "llama3.2:3b") -> str:
        """
        Generates an educational breakdown. Uses local Ollama LLM if reachable;
        otherwise provides deep structured domain logic.
        """
        headline = sim_result["headline"]
        stock = sim_result["primary_stock"]
        sector = sim_result["sector"]
        ret = sim_result["estimated_price_change_pct"]
        direction = sim_result["projected_direction"]
        
        # Try local Ollama LLM if enabled
        if use_ollama:
            prompt = (
                f"You are a seasoned financial market mentor teaching people how news affects stock prices.\n"
                f"News Event: \"{headline}\"\n"
                f"Simulation Data:\n"
                f"- Affected Stock: {stock} ({sector})\n"
                f"- Simulated Movement: {direction} ({ret}% projected next-day change)\n"
                f"- Peer Spillover: {sim_result['peer_sector_spillover']}\n\n"
                f"Please explain clearly in 3 concise paragraphs:\n"
                f"1. What is happening: Why does this news impact {stock} directly?\n"
                f"2. Market Mechanism: What are institutional investors, fund managers, and retail traders reacting to (e.g. uncertainty, earnings, guidance)?\n"
                f"3. Spillover & Educational Lesson: What should a market observer watch out for in the coming days?"
            )
            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": ollama_model, "prompt": prompt, "stream": False},
                    timeout=8
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "").strip()
            except Exception:
                pass  # Fallback to local expert synthesis

        # High-quality offline explanation generator
        explanation = (
            f"### Educational Breakdown & Market Dynamics:\n"
            f"1. **Direct Company Impact ({stock})**:\n"
            f"   The news implies a **{direction}** bias. In financial markets, news events involving corporate leadership, "
            f"   financial health, or legal scrutiny alter the risk premium required by investors. "
            f"   Our quantitative model projects an immediate next-session move of **{ret}%** with an intraday volatility range of ~**{sim_result['expected_intraday_swing_pct']}%**.\n\n"
            f"2. **The Underlying Mechanism**:\n"
            f"   - **Uncertainty Discount**: Markets dislike ambiguity. When unforeseen headlines break, foreign and domestic funds temporarily pause buying or trim positions to protect portfolios.\n"
            f"   - **Sector Contagion**: Related firms ({', '.join(sim_result['peer_sector_spillover'].keys())}) experience secondary sympathy moves ({sim_result['peer_sector_spillover']}) because institutional algorithms trade sector baskets as a cohesive group.\n\n"
            f"3. **What to Learn / Watch**:\n"
            f"   Over the next 1-3 trading sessions, watch for clarification notices to stock exchanges (BSE/NSE) and whether trading volume stabilizes. Often, initial reactive spikes provide contrarian equilibrium once facts are digested."
        )
        return explanation

def run_cli():
    print("=" * 68)
    print("   CrafTrade: Financial News & Market Impact Simulator (Local)")
    print("=" * 68)
    print("[*] Loading local simulation model...")
    try:
        simulator = MarketSimulator()
        print("[+] Simulator ready! Type your news headline to see what happens.")
        print("    Example: 'today tcs ceo resigned'")
        print("    Example: 'infosys reports 18% jump in quarterly net profit'")
        print("    (Type 'quit' or 'exit' to stop)\n")
    except Exception as e:
        print(f"[!] Error: {e}")
        print("    Please ensure you ran: python data_pipeline/build_simulation_dataset.py")
        print("    and: python models/train_local_simulator.py")
        sys.exit(1)

    while True:
        try:
            headline = input("\nEnter news headline: ").strip()
            if not headline:
                continue
            if headline.lower() in ['exit', 'quit', 'q']:
                print("Exiting CrafTrade Simulator. Goodbye!")
                break
                
            sim = simulator.simulate_event(headline)
            
            print("\n" + "-" * 60)
            print(f"📊 SIMULATION RESULTS for: \"{headline}\"")
            print("-" * 60)
            print(f"🎯 Target Stock Affected : {sim['primary_stock']} ({sim['sector']})")
            print(f"📈 Directional Bias      : {sim['projected_direction']}")
            print(f"📉 Estimated Price Delta : {sim['estimated_price_change_pct']}%")
            print(f"⚡ Intraday Volatility    : ±{sim['expected_intraday_swing_pct']}%")
            print(f"🌐 Sector Spillover      : {sim['peer_sector_spillover']}")
            print("-" * 60)
            
            explanation = simulator.get_educational_explanation(sim)
            print(explanation)
            print("-" * 60)
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    run_cli()

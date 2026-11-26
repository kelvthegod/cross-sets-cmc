# Cross Sets — CoinMarketCap Cross-Market Intelligence & Analyze Platform

**Hackathon Track:** Data and Visualisation (`Build with CMC: API Hackathon, Round 2`)  
**Stack:** `FastAPI` + `Polars` + `httpx` + `TradingView Lightweight Charts`

## Overview
Cross Sets enables traders and researchers to analyze relationships across CoinMarketCap's live cryptocurrency markets, sector categories, DEX liquidity pools, and the CoinMarketCap 20 Index on one comparable scale—without spreadsheets or coding.

### Key Features
1. **Embedded Home "Analyze" Workbench & Unified Desk (`/dashboard`):** Compare any two CMC-listed assets across all **Four Relationship Models** (`01 Relative Performance`, `02 Ratio Model`, `03 Difference Model`, `04 Beta-Adjusted Model`) with live TradingView Lightweight Charts and Polars performance statistics.
2. **CoinMarketCap Markets Board:** Live ranked CMC spot markets with 1h/24h/7d momentum, volume, market cap, and dominance.
3. **CMC Market Heatmaps:** Interactive sector-level (`/v1/cryptocurrency/categories`) and top-asset market cap heatmaps showing 24h capital rotation.
4. **CEX-vs-DEX Liquidity Basis & Market-Neutral Residual Alpha:** Regresses pair performance against `/v3/index/cmc20-historical` and tracks basis spreads between CEX spot quotes and `/v4/dex/spot-pairs/latest`.

## Security & GitHub Hygiene
- API keys are loaded strictly from `.env` via `python-dotenv` and excluded from version control via `.gitignore`.
- Never commit `.env` to GitHub. Only `.env.example` is tracked.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your CMC_API_KEY inside .env
uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000` for the landing page + embedded Analyze workbench, or `http://localhost:8000/dashboard` for the unified full-screen desk.

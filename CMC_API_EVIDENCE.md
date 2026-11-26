# CoinMarketCap API Integration Evidence & Product Feedback

## CoinMarketCap Endpoints Used
1. `GET /v1/cryptocurrency/map` — Canonical numeric `crypto_id` lookup (0 credits).
2. `GET /v1/cryptocurrency/listings/latest` — Live CMC Markets board, dominance, and 1h/24h/7d changes.
3. `GET /v1/cryptocurrency/categories` — Live CMC Market Sector Heatmaps (`Layer 1`, `Smart Contracts`, `DeFi`, `AI & Big Data`, `Memes`, `Layer 2`).
4. `GET /v1/global-metrics/quotes/latest` — Global crypto market cap, 24h volume, and BTC/ETH dominance metrics.
5. `GET /v3/cryptocurrency/quotes/historical` — Aligned historical daily time series for Polars multi-model vectorization.
6. `GET /v3/index/cmc20-historical` — CoinMarketCap 20 Index historical benchmark for OLS beta and residual alpha calculations.
7. `GET /v4/dex/spot-pairs/latest` — On-chain DEX pool liquidity and CEX-vs-DEX basis comparisons.

## API Feedback
- **What worked well:** Combining `/v1/cryptocurrency/categories`, `/v3/cryptocurrency/quotes/historical`, and `/v3/index/cmc20-historical` under one API key allows Cross Sets to compute sector heatmaps, pair ratios, and index-neutral beta regressions in a single Polars pipeline.
- **Where it got in the way:** DEX endpoints (`/v4/dex/spot-pairs/latest`) key pools by contract address and network slug rather than `crypto_id`, requiring symbol normalization when joining DEX execution quotes with CEX listings.

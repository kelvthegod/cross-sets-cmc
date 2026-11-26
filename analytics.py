"""
Polars Quantitative & Visualisation Engine for Cross Sets (CoinMarketCap Markets Only).
Computes:
  1. CMC Markets Board (/v1/cryptocurrency/listings/latest)
  2. CMC Market Heatmaps — Sector Categories (/v1/cryptocurrency/categories) & Top Crypto Heatmap
  3. Four Relationship Models & Performance Analytics (/v3/cryptocurrency/quotes/historical + /v3/index/cmc20-historical):
     - Model 01: Relative Performance (A vs B, rebased = 100)
     - Model 02: Ratio Model (A / B)
     - Model 03: Difference Model (A - B)
     - Model 04: Beta-Adjusted Model (A - beta * B, plus Market Residual Alpha vs CMC 20 Index)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import polars as pl


def build_cmc_markets_board(listings_data: List[Dict[str, Any]], currency: str = "USD") -> List[Dict[str, Any]]:
    """Transforms /v1/cryptocurrency/listings/latest into a ranked Polars market table."""
    rows: List[Dict[str, Any]] = []
    for item in listings_data:
        quote = (item.get("quote") or {}).get(currency, {})
        price = quote.get("price")
        if price is None:
            continue
        rows.append(
            {
                "id": int(item.get("id", 0)),
                "rank": int(item.get("cmc_rank", 999)),
                "name": item.get("name", ""),
                "symbol": item.get("symbol", ""),
                "price": float(price),
                "percent_change_1h": float(quote.get("percent_change_1h") or 0.0),
                "percent_change_24h": float(quote.get("percent_change_24h") or 0.0),
                "percent_change_7d": float(quote.get("percent_change_7d") or 0.0),
                "volume_24h": float(quote.get("volume_24h") or 0.0),
                "market_cap": float(quote.get("market_cap") or 0.0),
                "market_cap_dominance": float(quote.get("market_cap_dominance") or 0.0),
            }
        )

    if not rows:
        return []

    df = pl.DataFrame(rows).with_columns(
        [
            pl.when(pl.col("market_cap") > 0)
            .then((pl.col("volume_24h") / pl.col("market_cap")) * 100.0)
            .otherwise(0.0)
            .round(2)
            .alias("turnover_pct"),
            pl.col("price").round(4),
            pl.col("percent_change_1h").round(2),
            pl.col("percent_change_24h").round(2),
            pl.col("percent_change_7d").round(2),
            pl.col("market_cap_dominance").round(2),
        ]
    ).sort("rank")

    return df.to_dicts()


def build_cmc_sector_heatmap(categories_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms /v1/cryptocurrency/categories into normalized heatmap tiles
    sized by market_cap and colored by market_cap_change (24h).
    """
    rows: List[Dict[str, Any]] = []
    for cat in categories_data:
        mcap = float(cat.get("market_cap") or 0.0)
        vol = float(cat.get("volume") or 0.0)
        if mcap <= 0:
            continue
        rows.append(
            {
                "id": str(cat.get("id", "")),
                "name": cat.get("name", "Market Sector"),
                "title": cat.get("title", cat.get("name", "")),
                "num_tokens": int(cat.get("num_tokens") or 0),
                "market_cap": mcap,
                "market_cap_change_24h": round(float(cat.get("market_cap_change") or 0.0), 2),
                "volume_24h": vol,
                "volume_change_24h": round(float(cat.get("volume_change") or 0.0), 2),
                "avg_price_change_24h": round(float(cat.get("avg_price_change") or 0.0), 2),
            }
        )

    if not rows:
        return []

    df = pl.DataFrame(rows).sort("market_cap", descending=True).head(16)
    total_mcap = float(df["market_cap"].sum() or 1.0)
    df = df.with_columns(
        [
            ((pl.col("market_cap") / total_mcap) * 100.0).round(2).alias("weight_pct"),
        ]
    )
    return df.to_dicts()


def compute_dex_cex_basis(
    cex_listings: List[Dict[str, Any]],
    dex_pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compares CEX spot quotes against on-chain DEX pair quotes in Polars."""
    cex_map: Dict[str, float] = {}
    for item in cex_listings:
        sym = (item.get("symbol") or "").upper()
        p = item.get("price")
        if sym and p:
            cex_map[sym] = float(p)

    rows: List[Dict[str, Any]] = []
    for pair in dex_pairs:
        base_sym = (pair.get("base_asset_symbol") or "").upper()
        quote_obj = (
            pair["quote"][0]
            if isinstance(pair.get("quote"), list) and pair["quote"]
            else (pair.get("quote") or {}).get("USD", {})
        )
        dex_price = quote_obj.get("price")
        liquidity = quote_obj.get("liquidity") or 0.0
        vol_24h = quote_obj.get("volume_24h") or 0.0
        cex_price = cex_map.get(base_sym)

        if dex_price is not None and cex_price:
            rows.append(
                {
                    "pair": pair.get("name", f"{base_sym}/USDC"),
                    "network": pair.get("network_slug", "ethereum"),
                    "dex": pair.get("dex_slug", "uniswap-v3"),
                    "base_symbol": base_sym,
                    "cex_price": float(cex_price),
                    "dex_price": float(dex_price),
                    "liquidity_usd": float(liquidity),
                    "volume_24h": float(vol_24h),
                }
            )

    if not rows:
        return []

    df = pl.DataFrame(rows).with_columns(
        [
            (((pl.col("dex_price") - pl.col("cex_price")) / pl.col("cex_price")) * 10000.0)
            .round(2)
            .alias("basis_bps"),
        ]
    ).sort("liquidity_usd", descending=True)

    return df.to_dicts()


def compute_four_models_and_performance(
    timestamps: List[str],
    prices_a: List[float],
    prices_b: List[float],
    cmc20_series: List[float],
) -> Dict[str, Any]:
    """
    Computes all 4 Cross Sets relationship models + performance diagnostics in Polars:
      01 Relative Performance: Norm(A) - Norm(B) (Rebased to 100)
      02 Ratio Model:          Price(A) / Price(B)
      03 Difference Model:     Price(A) - Price(B)
      04 Beta-Adjusted Model:  Norm(A) - beta * Norm(B) (plus Residual Alpha vs CMC 20 Index)
    """
    n = min(len(timestamps), len(prices_a), len(prices_b), len(cmc20_series))
    if n < 3:
        raise ValueError("At least 3 aligned data points are required for relationship analysis.")

    df = pl.DataFrame(
        {
            "timestamp": timestamps[:n],
            "price_a": [float(x) for x in prices_a[:n]],
            "price_b": [float(x) for x in prices_b[:n]],
            "cmc20": [float(x) for x in cmc20_series[:n]],
        }
    )

    base_a = df["price_a"][0] or 1.0
    base_b = df["price_b"][0] or 1.0
    base_idx = df["cmc20"][0] or 1.0

    df = df.with_columns(
        [
            (pl.col("price_a") / base_a * 100.0).alias("norm_a"),
            (pl.col("price_b") / base_b * 100.0).alias("norm_b"),
            (pl.col("cmc20") / base_idx * 100.0).alias("norm_cmc20"),
            pl.when(pl.col("price_b") != 0).then(pl.col("price_a") / pl.col("price_b")).otherwise(0.0).alias("ratio_ab"),
            (pl.col("price_a") - pl.col("price_b")).alias("diff_ab"),
            pl.col("price_a").pct_change().fill_null(0.0).alias("ret_a"),
            pl.col("price_b").pct_change().fill_null(0.0).alias("ret_b"),
            pl.col("cmc20").pct_change().fill_null(0.0).alias("ret_idx"),
        ]
    )

    # OLS Beta of Asset A against Asset B
    cov_ab = df.select(pl.cov("ret_a", "ret_b")).item() or 0.0
    var_b = df.select(pl.var("ret_b")).item() or 0.0
    beta_ab = round(cov_ab / var_b, 4) if var_b > 1e-12 else 1.0

    # OLS Beta of Asset A against CMC 20 Market Index
    cov_a_idx = df.select(pl.cov("ret_a", "ret_idx")).item() or 0.0
    var_idx = df.select(pl.var("ret_idx")).item() or 0.0
    beta_cmc20 = round(cov_a_idx / var_idx, 4) if var_idx > 1e-12 else 1.0

    df = df.with_columns(
        [
            (pl.col("norm_a") - pl.col("norm_b")).round(4).alias("rel_perf"),
            (pl.col("norm_a") - (beta_ab * pl.col("norm_b"))).round(4).alias("beta_adjusted"),
            (pl.col("norm_a") - (beta_cmc20 * pl.col("norm_cmc20"))).round(4).alias("cmc20_residual_alpha"),
        ]
    )

    # Performance metrics for Asset A vs Asset B
    def _asset_perf(price_col: str, ret_col: str) -> Dict[str, Any]:
        start_p = float(df[price_col][0])
        end_p = float(df[price_col][-1])
        total_ret = ((end_p - start_p) / max(start_p, 1e-9)) * 100.0
        daily_std = float(df[ret_col].std() or 0.0)
        ann_vol = daily_std * math.sqrt(365.0) * 100.0
        peak = df[price_col].cum_max()
        dd = ((df[price_col] - peak) / peak * 100.0).min() or 0.0
        sharpe_proxy = (total_ret / max(ann_vol, 1e-6))
        return {
            "start_price": round(start_p, 4),
            "latest_price": round(end_p, 4),
            "period_return_pct": round(total_ret, 2),
            "annualized_vol_pct": round(ann_vol, 2),
            "max_drawdown_pct": round(float(dd), 2),
            "return_vol_ratio": round(sharpe_proxy, 2),
        }

    perf_a = _asset_perf("price_a", "ret_a")
    perf_b = _asset_perf("price_b", "ret_b")

    ratio_mean = float(df["ratio_ab"].mean() or 0.0)
    ratio_std = float(df["ratio_ab"].std() or 1e-9)
    latest_ratio = float(df["ratio_ab"][-1])
    z_score = round((latest_ratio - ratio_mean) / max(ratio_std, 1e-9), 3)

    corr_val = df.select(pl.corr("ret_a", "ret_b")).item()
    corr_clean = 0.0 if (corr_val is None or math.isnan(corr_val)) else round(float(corr_val), 4)

    conviction = min(99, max(15, int(round(min(abs(z_score) * 26.0, 62.0) + abs(corr_clean) * 35.0))))

    series_records = df.select(
        [
            "timestamp",
            pl.col("price_a").round(4),
            pl.col("price_b").round(4),
            pl.col("norm_a").round(3),
            pl.col("norm_b").round(3),
            pl.col("norm_cmc20").round(3),
            pl.col("rel_perf"),
            pl.col("ratio_ab").round(6),
            pl.col("diff_ab").round(4),
            pl.col("beta_adjusted"),
            pl.col("cmc20_residual_alpha"),
        ]
    ).to_dicts()

    return {
        "summary": {
            "rel_perf_spread_pts": round(float(df["rel_perf"][-1]), 3),
            "latest_ratio": round(latest_ratio, 6),
            "latest_diff": round(float(df["diff_ab"][-1]), 4),
            "beta_ab": beta_ab,
            "beta_cmc20": beta_cmc20,
            "beta_adjusted_pts": round(float(df["beta_adjusted"][-1]), 3),
            "cmc20_residual_alpha_pts": round(float(df["cmc20_residual_alpha"][-1]), 3),
            "ratio_z_score": z_score,
            "correlation": corr_clean,
            "conviction_score": conviction,
        },
        "performance": {
            "asset_a": perf_a,
            "asset_b": perf_b,
        },
        "series": series_records,
    }

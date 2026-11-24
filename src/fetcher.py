import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import ALL_TICKERS, CACHE_EXPIRY_HOURS, DATA_RAW_DIR

logger = logging.getLogger(__name__)


def _is_cache_valid(
    cache_path: Path,
    expiry_hours: int = CACHE_EXPIRY_HOURS,
) -> bool:
    """Return whether a cache file exists and is within the allowed age."""
    try:
        age_seconds = time.time() - cache_path.stat().st_mtime
    except OSError:
        return False

    return 0 <= age_seconds < expiry_hours * 3600


def _to_json_value(value: Any) -> Any:
    """Convert common pandas/numpy values into JSON-safe Python values."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _df_to_clean_dict(df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Convert a financial statement DataFrame into:
    { "YYYY-MM-DD": { "Line Item": value } }
    """
    if df is None or df.empty:
        return {}

    cleaned: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
        cleaned[date_str] = {
            str(index): _to_json_value(value)
            for index, value in df[col].items()
        }

    return cleaned


def _get_statement(stock: yf.Ticker, attribute: str, ticker: str) -> Dict[str, Dict[str, Any]]:
    """Fetch and clean one statement, logging errors without failing the ticker."""
    try:
        statement = getattr(stock, attribute)
        return _df_to_clean_dict(statement)
    except Exception as exc:
        logger.warning("[%s] Could not retrieve %s: %s", ticker, attribute, exc)
        return {}


def _load_cache(cache_file: Path, ticker: str) -> Optional[Dict[str, Any]]:
    """Load valid JSON cache data, returning None if it cannot be used."""
    try:
        with cache_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            return payload
        logger.warning("[%s] Cache contents are not a JSON object; refreshing", ticker)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[%s] Could not read cache; refreshing: %s", ticker, exc)

    return None


def _save_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid leaving a partial cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def fetch_ticker_payload(
    ticker: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Fetch financial statements and market data, using a local JSON cache."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty")

    cache_file = DATA_RAW_DIR / f"{ticker}_raw.json"

    if not force_refresh and _is_cache_valid(cache_file):
        cached_payload = _load_cache(cache_file, ticker)
        if cached_payload is not None:
            logger.info("[%s] Loaded cached payload from %s", ticker, cache_file.name)
            return cached_payload

    logger.info("[%s] Fetching live financial data from yfinance", ticker)
    stock = yf.Ticker(ticker)

    try:
        info = stock.info or {}
    except Exception as exc:
        logger.warning("[%s] Could not retrieve stock.info: %s", ticker, exc)
        info = {}

    raw_snapshot = {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": (
            info.get("currentPrice")
            if info.get("currentPrice") is not None
            else info.get("regularMarketPrice")
        ),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "beta": info.get("beta"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "ev_to_revenue": info.get("enterpriseToRevenue"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
    }

    string_keys = {"ticker", "company_name", "sector", "industry"}
    market_snapshot = {
        key: (val if key in string_keys else _to_json_value(val))
        for key, val in raw_snapshot.items()
    }

    payload: Dict[str, Any] = {
        "ticker": ticker,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_snapshot": market_snapshot,
        "income_statement": _get_statement(stock, "financials", ticker),
        "balance_sheet": _get_statement(stock, "balance_sheet", ticker),
        "cash_flow": _get_statement(stock, "cashflow", ticker),
    }

    _save_json_atomic(cache_file, payload)
    logger.info("[%s] Saved payload to %s", ticker, cache_file.name)
    return payload


def fetch_all_companies(
    tickers: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch payloads for the requested tickers, continuing after failures."""
    target_list = ALL_TICKERS if tickers is None else tickers
    results: Dict[str, Dict[str, Any]] = {}

    for raw_ticker in target_list:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            logger.warning("Skipping an empty ticker")
            continue

        try:
            results[ticker] = fetch_ticker_payload(
                ticker,
                force_refresh=force_refresh,
            )
        except Exception:
            logger.exception("[%s] Failed to fetch data", ticker)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data = fetch_all_companies(force_refresh=False)

    for ticker, content in data.items():
        snapshot = content.get("market_snapshot", {})
        fiscal_years = list(content.get("income_statement", {}).keys())

        print(
            f"{ticker}: {snapshot.get('company_name')} | "
            f"Price: ${snapshot.get('current_price')} | "
            f"Fiscal Years: {fiscal_years}"
        )

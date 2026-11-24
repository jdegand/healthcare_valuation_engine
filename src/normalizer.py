import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DCF_DEFAULTS, TARGET_TICKER

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float; return None for missing/invalid values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return result if math.isfinite(result) else None


def _normalized_key(value: Any) -> str:
    """Normalize statement labels for tolerant alias matching."""
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _get_line_item(
    statement_year: Any,
    candidates: List[str],
    default: Optional[float] = 0.0,
) -> Optional[float]:
    """Return the first matching line item, tolerating case and punctuation differences."""
    if not isinstance(statement_year, dict):
        return default

    # Preserve exact matches as the preferred lookup.
    for key in candidates:
        if key in statement_year:
            value = _to_float(statement_year[key])
            if value is not None:
                return value

    normalized_items = {
        _normalized_key(key): value
        for key, value in statement_year.items()
    }

    for candidate in candidates:
        value = _to_float(normalized_items.get(_normalized_key(candidate)))
        if value is not None:
            return value

    return default


def _get_required_line_item(
    statement_year: Any,
    candidates: List[str],
) -> Tuple[Optional[float], bool]:
    """Return a line item and whether a usable value was found."""
    value = _get_line_item(statement_year, candidates, default=None)
    return value, value is not None


def _clean_date(value: Any) -> Optional[str]:
    """Normalize a fiscal-date key to YYYY-MM-DD."""
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp.strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_company_historicals(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalize annual financial statements for one company.

    Missing NWC components are represented as NaN instead of being treated as zero,
    preventing incomplete balance sheets from creating misleading UFCF calculations.
    """
    ticker = str(payload.get("ticker", "UNKNOWN")).strip().upper()
    income = payload.get("income_statement") or {}
    balance_sheet = payload.get("balance_sheet") or {}
    cash_flow = payload.get("cash_flow") or {}

    if not isinstance(income, dict):
        raise ValueError(f"[{ticker}] income_statement must be a dictionary.")

    valid_dates: List[str] = []
    for raw_date, year_data in income.items():
        date = _clean_date(raw_date)
        revenue = _get_line_item(
            year_data,
            ["Total Revenue", "Operating Revenue"],
            default=None,
        )
        if date is not None and revenue is not None and revenue > 0:
            valid_dates.append(date)

    valid_dates = sorted(set(valid_dates))
    if not valid_dates:
        raise ValueError(f"[{ticker}] No fiscal years with positive revenue found.")

    # Index statements by normalized date, so date-key formatting differences don't
    # prevent joining their annual data.
    def index_by_date(statement: Any) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        if not isinstance(statement, dict):
            return result

        for raw_date, values in statement.items():
            date = _clean_date(raw_date)
            if date and isinstance(values, dict):
                result[date] = values
        return result

    income_by_date = index_by_date(income)
    balance_by_date = index_by_date(balance_sheet)
    cash_flow_by_date = index_by_date(cash_flow)

    rows: List[Dict[str, Any]] = []

    for date in valid_dates:
        inc_year = income_by_date.get(date, {})
        bs_year = balance_by_date.get(date, {})
        cf_year = cash_flow_by_date.get(date, {})

        revenue = _get_line_item(
            inc_year, ["Total Revenue", "Operating Revenue"], default=None
        )
        if revenue is None or revenue <= 0:
            continue

        ebit = _get_line_item(
            inc_year, ["EBIT", "Operating Income"], default=0.0
        ) or 0.0

        # Use standard EBITDA when available; calculate it from EBIT + D&A otherwise.
        ebitda_value = _get_line_item(
            inc_year, ["EBITDA"], default=None
        )
        normalized_ebitda = _get_line_item(
            inc_year, ["Normalized EBITDA"], default=None
        )

        dna = _get_line_item(
            cf_year,
            [
                "Depreciation And Amortization",
                "Depreciation & Amortization",
                "Reconciled Depreciation",
                "Depreciation",
            ],
            default=None,
        )
        if dna is None:
            dna = _get_line_item(
                inc_year,
                ["Depreciation And Amortization", "Reconciled Depreciation"],
                default=0.0,
            ) or 0.0

        if ebitda_value is None:
            ebitda = ebit + dna
        else:
            ebitda = ebitda_value

        if normalized_ebitda is None:
            normalized_ebitda = ebitda

        pretax_income = _get_line_item(
            inc_year, ["Pretax Income", "Income Before Tax"], default=None
        )
        tax_provision = _get_line_item(
            inc_year, ["Tax Provision", "Income Tax Expense"], default=None
        )

        if (
            pretax_income is not None
            and pretax_income > 0
            and tax_provision is not None
            and tax_provision > 0
        ):
            tax_rate = float(np.clip(tax_provision / pretax_income, 0.0, 0.40))
        else:
            tax_rate = float(DCF_DEFAULTS["default_tax_rate"])

        # CapEx is represented as a positive cash outflow.
        raw_capex = _get_line_item(
            cf_year, ["Capital Expenditure", "Capital Expenditures"], default=None
        )
        capex = abs(raw_capex) if raw_capex is not None else np.nan
        if raw_capex is not None and raw_capex > 0:
            logger.debug(
                "[%s] Positive source CapEx value for %s; using its absolute value",
                ticker,
                date,
            )

        # Calculate non-cash NWC only when all required inputs are available.
        current_assets, has_current_assets = _get_required_line_item(
            bs_year, ["Current Assets", "Total Current Assets"]
        )
        cash_equivalents, has_cash = _get_required_line_item(
            bs_year,
            [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents And Short Term Investments",
            ],
        )
        current_liabilities, has_current_liabilities = _get_required_line_item(
            bs_year, ["Current Liabilities", "Total Current Liabilities"]
        )
        short_term_debt, has_short_debt = _get_required_line_item(
            bs_year,
            ["Current Debt", "Current Debt And Capital Lease Obligation"],
        )

        if has_current_assets and has_cash and has_current_liabilities and has_short_debt:
            nwc = (
                (current_assets - cash_equivalents)
                - (current_liabilities - short_term_debt)
            )
        else:
            nwc = np.nan
            logger.debug(
                "[%s] Incomplete balance-sheet NWC inputs for %s",
                ticker,
                date,
            )

        total_debt = _get_line_item(bs_year, ["Total Debt"], default=np.nan)
        minority_interest = _get_line_item(
            bs_year, ["Minority Interest", "Minority Interest And Preferred Stock"],
            default=np.nan,
        )

        rows.append(
            {
                "fiscal_date": date,
                "fiscal_year": date[:4],
                "revenue": revenue,
                "ebitda": ebitda,
                "normalized_ebitda": normalized_ebitda,
                "ebit": ebit,
                "tax_rate": tax_rate,
                "dna": dna,
                "capex": capex,
                "nwc": nwc,
                "total_debt": total_debt,
                "cash": cash_equivalents if has_cash else np.nan,
                "minority_interest": minority_interest,
            }
        )

    df = pd.DataFrame(rows).sort_values("fiscal_date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"[{ticker}] No usable historical rows were produced.")

    # Historical growth and operating ratios.
    df["revenue_growth"] = df["revenue"].pct_change()
    df["ebitda_margin"] = df["ebitda"] / df["revenue"]
    df["ebit_margin"] = df["ebit"] / df["revenue"]
    df["dna_pct_rev"] = df["dna"] / df["revenue"]
    df["capex_pct_rev"] = df["capex"] / df["revenue"]

    # Keep the first delta at zero. Missing NWC values remain missing.
    df["delta_nwc"] = df["nwc"].diff()
    if not df.empty and pd.notna(df.loc[df.index[0], "nwc"]):
        df.loc[df.index[0], "delta_nwc"] = 0.0

    df["nopat"] = df["ebit"] * (1.0 - df["tax_rate"])
    df["ufcf"] = df["nopat"] + df["dna"] - df["capex"] - df["delta_nwc"]

    return df


def _snapshot_number(snapshot: Dict[str, Any], key: str) -> Optional[float]:
    """Safely parse a numeric market snapshot value."""
    return _to_float(snapshot.get(key))


def build_comps_table(
    all_payloads: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a trading-comps table. Invalid peer payloads are logged and skipped.

    EV uses market cap + debt + minority interest - cash when market cap is
    available; otherwise it falls back to the supplied enterprise value.
    """
    comps_rows: List[Dict[str, Any]] = []
    target_ticker = str(TARGET_TICKER).strip().upper()

    for raw_ticker, payload in all_payloads.items():
        ticker = str(raw_ticker).strip().upper()

        try:
            snapshot = payload.get("market_snapshot") or {}
            if not isinstance(snapshot, dict):
                logger.warning("[%s] Invalid market_snapshot; skipping", ticker)
                continue

            history = normalize_company_historicals(payload)
            latest = history.iloc[-1]

            price = _snapshot_number(snapshot, "current_price")
            market_cap = _snapshot_number(snapshot, "market_cap")
            snapshot_debt = _snapshot_number(snapshot, "total_debt")
            snapshot_cash = _snapshot_number(snapshot, "total_cash")

            historical_debt = _to_float(latest["total_debt"])
            historical_cash = _to_float(latest["cash"])
            minority_interest = _to_float(latest["minority_interest"])

            debt = (
                snapshot_debt
                if snapshot_debt is not None
                else historical_debt
            )
            cash = snapshot_cash if snapshot_cash is not None else historical_cash

            # Missing values are not silently assumed to be zero.
            debt_for_ev = debt if debt is not None else 0.0
            cash_for_ev = cash if cash is not None else 0.0
            minority_for_ev = minority_interest if minority_interest is not None else 0.0

            snapshot_ev = _snapshot_number(snapshot, "enterprise_value")
            if market_cap is not None and market_cap > 0:
                enterprise_value = (
                    market_cap + debt_for_ev + minority_for_ev - cash_for_ev
                )
            else:
                enterprise_value = snapshot_ev

            revenue = _to_float(latest["revenue"])
            ebitda = _to_float(latest["ebitda"])
            ev_to_revenue = (
                enterprise_value / revenue
                if enterprise_value is not None
                and enterprise_value > 0
                and revenue is not None
                and revenue > 0
                else np.nan
            )
            ev_to_ebitda = (
                enterprise_value / ebitda
                if enterprise_value is not None
                and enterprise_value > 0
                and ebitda is not None
                and ebitda > 0
                else np.nan
            )

            trailing_pe_raw = _snapshot_number(snapshot, "trailing_pe")
            forward_pe_raw = _snapshot_number(snapshot, "forward_pe")
            trailing_pe = (
                trailing_pe_raw
                if trailing_pe_raw is not None and trailing_pe_raw > 0
                else np.nan
            )
            forward_pe = (
                forward_pe_raw
                if forward_pe_raw is not None and forward_pe_raw > 0
                else np.nan
            )

            shares_outstanding = _snapshot_number(snapshot, "shares_outstanding")
            beta = _snapshot_number(snapshot, "beta")

            comps_rows.append(
                {
                    "ticker": ticker,
                    "company_name": snapshot.get("company_name") or ticker,
                    "is_target": ticker == target_ticker,
                    "share_price": price if price is not None else np.nan,
                    "market_cap": market_cap if market_cap is not None else np.nan,
                    "total_debt": debt if debt is not None else np.nan,
                    "minority_interest": (
                        minority_interest
                        if minority_interest is not None
                        else np.nan
                    ),
                    "cash": cash if cash is not None else np.nan,
                    "net_debt": (
                        debt - cash
                        if debt is not None and cash is not None
                        else np.nan
                    ),
                    "enterprise_value": (
                        enterprise_value
                        if enterprise_value is not None
                        else np.nan
                    ),
                    "ltm_revenue": revenue if revenue is not None else np.nan,
                    "ltm_ebitda": ebitda if ebitda is not None else np.nan,
                    "ebitda_margin": latest["ebitda_margin"],
                    "ev_to_revenue": ev_to_revenue,
                    "ev_to_ebitda": ev_to_ebitda,
                    "trailing_pe": trailing_pe,
                    "forward_pe": forward_pe,
                    "beta": beta if beta is not None else np.nan,
                    "shares_outstanding": (
                        shares_outstanding
                        if shares_outstanding is not None
                        else np.nan
                    ),
                }
            )

        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("[%s] Skipping comps row: %s", ticker, exc)
        except Exception:
            logger.exception("[%s] Unexpected error building comps row", ticker)

    return pd.DataFrame(comps_rows)


if __name__ == "__main__":
    from src.fetcher import fetch_all_companies

    raw_data = fetch_all_companies(force_refresh=False)
    target_ticker = str(TARGET_TICKER).strip().upper()

    if target_ticker in raw_data:
        target_history = normalize_company_historicals(raw_data[target_ticker])
        print(f"\n--- {target_ticker} Normalized Historicals ---")
        print(
            target_history[
                [
                    "fiscal_year",
                    "revenue",
                    "ebitda",
                    "capex",
                    "delta_nwc",
                    "ufcf",
                ]
            ]
        )
    else:
        logger.error("Target ticker %s was not fetched successfully", target_ticker)

    comps = build_comps_table(raw_data)
    print("\n--- Trading Comps ---")
    if not comps.empty:
        print(
            comps[
                [
                    "ticker",
                    "enterprise_value",
                    "ev_to_revenue",
                    "ev_to_ebitda",
                    "forward_pe",
                ]
            ]
        )
    else:
        print("No valid comps rows were produced.")
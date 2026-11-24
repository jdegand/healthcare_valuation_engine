import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import DCF_DEFAULTS, TARGET_TICKER

logger = logging.getLogger(__name__)


def _finite_float(value: Any, name: str) -> float:
    """Convert a value to a finite float or raise a descriptive error."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")

    return result


def _is_missing_scalar(value: Any) -> bool:
    """Safely check whether a scalar is None, NaN, or pd.NA."""
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except (TypeError, ValueError):
        return False


def _required_row_value(row: pd.Series, key: str, row_name: str) -> float:
    """Extract a required numeric field from a Series."""
    if key not in row.index:
        raise ValueError(f"{row_name} is missing required field {key!r}")
    return _finite_float(row[key], f"{row_name}.{key}")


def _optional_row_value(
    row: pd.Series,
    key: str,
    row_name: str,
    default: float = 0.0,
) -> float:
    """Extract an optional numeric field, logging when the default is used."""
    default = _finite_float(default, f"{row_name}.{key} default")

    if key not in row.index:
        logger.info("%s.%s missing; defaulting to %s", row_name, key, default)
        return default

    value = row[key]
    if not np.isscalar(value) and not _is_missing_scalar(value):
        raise ValueError(f"{row_name}.{key} must be a scalar value")

    if _is_missing_scalar(value):
        logger.info("%s.%s null; defaulting to %s", row_name, key, default)
        return default

    return _finite_float(value, f"{row_name}.{key}")


def _target_flag(value: Any) -> bool:
    """Parse common boolean representations without ambiguous truthiness."""
    if _is_missing_scalar(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
        raise ValueError(f"Unrecognized is_target value: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    raise ValueError(f"Unrecognized is_target value: {value!r}")


def _get_target_row(comps_df: pd.DataFrame) -> pd.Series:
    """Return the configured target row, or raise if missing or mismatched."""
    if "is_target" not in comps_df.columns:
        raise ValueError("comps_df must contain an 'is_target' column")

    expected_ticker = str(TARGET_TICKER).strip().upper()
    target_rows = comps_df[comps_df["is_target"].map(_target_flag)]

    if target_rows.empty:
        raise ValueError(f"Target {expected_ticker!r} was not found in comps_df")

    if "ticker" in target_rows.columns:
        matched = target_rows[
            target_rows["ticker"].astype(str).str.strip().str.upper()
            == expected_ticker
        ]
        if matched.empty:
            found = target_rows["ticker"].tolist()
            raise ValueError(
                f"is_target rows {found!r} do not match "
                f"TARGET_TICKER {expected_ticker!r}"
            )
        target_rows = matched

    if len(target_rows) > 1:
        logger.warning("Multiple target rows found for %s; using the first", expected_ticker)

    return target_rows.iloc[0]


def _mean_last_valid(series: pd.Series, name: str) -> float:
    """Compute the mean of up to the last three finite observations."""
    values = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        raise ValueError(f"No valid historical values available for {name}")
    return float(values.iloc[-3:].mean())


def _safe_positive_median(series: pd.Series, name: str) -> float:
    """Compute the median of strictly positive finite observations."""
    values = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    values = values[values > 0]
    if values.empty:
        raise ValueError(f"No positive observations available for {name}")
    return float(values.median())


def _validated_projection_years(value: Any) -> int:
    """Require a positive whole-number projection horizon."""
    numeric = _finite_float(value, "projection_years")
    if not numeric.is_integer() or numeric < 1:
        raise ValueError("projection_years must be a positive whole number")
    return int(numeric)


def calculate_wacc(
    target_row: pd.Series,
    latest_hist: pd.Series,
) -> Dict[str, float]:
    """
    Compute WACC using CAPM for cost of equity and market-value capital weights.
    WACC weights include equity and debt only, per Wall Street convention.
    """
    rf = _finite_float(DCF_DEFAULTS["risk_free_rate"], "risk_free_rate")
    erp = _finite_float(DCF_DEFAULTS["equity_risk_premium"], "equity_risk_premium")

    beta_value = pd.to_numeric(
        pd.Series([target_row.get("beta")]), errors="coerce"
    ).iloc[0]
    beta = (
        float(beta_value)
        if pd.notna(beta_value) and math.isfinite(float(beta_value))
        else 1.10
    )

    cost_of_equity = rf + beta * erp
    kd_pretax = _finite_float(
        DCF_DEFAULTS["cost_of_debt_pretax"], "cost_of_debt_pretax"
    )

    tax_rate = _required_row_value(latest_hist, "tax_rate", "latest_hist")
    tax_rate = float(np.clip(tax_rate, 0.0, 1.0))
    kd_aftertax = kd_pretax * (1.0 - tax_rate)

    equity_val = _required_row_value(target_row, "market_cap", "target_row")
    debt_val = _optional_row_value(target_row, "total_debt", "target_row")

    if equity_val < 0 or debt_val < 0:
        raise ValueError("Market capitalization and debt must not be negative")

    total_cap = equity_val + debt_val
    if total_cap <= 0:
        raise ValueError("Cannot calculate WACC with zero total capital")

    weight_equity = equity_val / total_cap
    weight_debt = debt_val / total_cap
    wacc = weight_equity * cost_of_equity + weight_debt * kd_aftertax

    if not math.isfinite(wacc):
        raise ValueError("Calculated WACC is not finite")

    return {
        "risk_free_rate": rf,
        "equity_risk_premium": erp,
        "beta": beta,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt_pretax": kd_pretax,
        "effective_tax_rate": tax_rate,
        "cost_of_debt_aftertax": kd_aftertax,
        "market_cap": equity_val,
        "total_debt": debt_val,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "wacc": wacc,
    }


def _validate_assumptions(assumptions: Dict[str, Any]) -> Dict[str, Any]:
    """Return a validated copy of assumptions without mutating the input."""
    required = {
        "projection_years",
        "revenue_growth",
        "ebitda_margin",
        "dna_pct_rev",
        "capex_pct_rev",
        "nwc_pct_delta_rev",
        "tax_rate",
        "terminal_growth_rate",
        "exit_ebitda_multiple",
    }
    missing = required.difference(assumptions)
    if missing:
        raise ValueError(f"Missing DCF assumptions: {sorted(missing)}")

    validated = dict(assumptions)
    validated["projection_years"] = _validated_projection_years(
        validated["projection_years"]
    )
    for key in required - {"projection_years"}:
        validated[key] = _finite_float(validated[key], f"assumptions.{key}")

    if validated["exit_ebitda_multiple"] <= 0:
        raise ValueError("exit_ebitda_multiple must be positive")
    if not 0.0 <= validated["terminal_growth_rate"] <= 0.15:
        raise ValueError("terminal_growth_rate must be between 0% and 15%")
    if not 0.0 <= validated["tax_rate"] <= 1.0:
        raise ValueError("tax_rate must be between 0 and 1")
    if validated["revenue_growth"] <= -1.0:
        raise ValueError("revenue_growth must be greater than -100%")
    if not -1.0 <= validated["ebitda_margin"] <= 1.0:
        raise ValueError("ebitda_margin must be between -100% and 100%")
    if not 0.0 <= validated["dna_pct_rev"] <= 1.0:
        raise ValueError("dna_pct_rev must be between 0% and 100%")
    if not 0.0 <= validated["capex_pct_rev"] <= 1.0:
        raise ValueError("capex_pct_rev must be between 0% and 100%")
    if not -0.50 <= validated["nwc_pct_delta_rev"] <= 0.50:
        raise ValueError("nwc_pct_delta_rev must be between -50% and 50%")

    return validated


def derive_dcf_assumptions(
    hist_df: pd.DataFrame,
    comps_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Derive baseline forecast assumptions from company history and peer multiples."""
    if hist_df.empty:
        raise ValueError("hist_df is empty")

    required_cols = {
        "revenue",
        "revenue_growth",
        "ebitda_margin",
        "dna_pct_rev",
        "capex_pct_rev",
        "delta_nwc",
        "tax_rate",
    }
    missing = required_cols.difference(hist_df.columns)
    if missing:
        raise ValueError(f"hist_df is missing columns: {sorted(missing)}")
    if "is_target" not in comps_df.columns:
        raise ValueError("comps_df must contain an 'is_target' column")

    growth = (
        pd.to_numeric(hist_df["revenue_growth"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if growth.empty:
        logger.warning("No valid historical revenue growth; using 5%% baseline growth")
        base_rev_growth = 0.05
    else:
        base_rev_growth = float(np.clip(float(growth.mean()) * 0.75, 0.04, 0.08))

    revenue = pd.to_numeric(hist_df["revenue"], errors="coerce")
    revenue_diffs = revenue.diff().replace([np.inf, -np.inf], np.nan)
    nwc_diffs = pd.to_numeric(hist_df["delta_nwc"], errors="coerce")
    nwc_ratios = (nwc_diffs / revenue_diffs).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    base_nwc_pct = (
        float(np.clip(float(nwc_ratios.median()), -0.02, 0.05))
        if not nwc_ratios.empty
        else 0.02
    )

    target_mask = comps_df["is_target"].map(_target_flag)
    peers_only = comps_df[~target_mask]
    peer_median_ebitda = _safe_positive_median(
        peers_only.get("ev_to_ebitda", pd.Series(dtype=float)),
        "peer EV/EBITDA",
    )

    tax_rates = (
        pd.to_numeric(hist_df["tax_rate"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if tax_rates.empty:
        raise ValueError("No valid historical tax rates")

    raw_assumptions: Dict[str, Any] = {
        "projection_years": _validated_projection_years(
            DCF_DEFAULTS["projection_years"]
        ),
        "revenue_growth": base_rev_growth,
        "ebitda_margin": _mean_last_valid(
            hist_df["ebitda_margin"], "EBITDA margin"
        ),
        "dna_pct_rev": _mean_last_valid(hist_df["dna_pct_rev"], "D&A / revenue"),
        "capex_pct_rev": _mean_last_valid(
            hist_df["capex_pct_rev"], "CapEx / revenue"
        ),
        "nwc_pct_delta_rev": base_nwc_pct,
        "tax_rate": float(np.clip(tax_rates.iloc[-1], 0.0, 1.0)),
        "terminal_growth_rate": _finite_float(
            DCF_DEFAULTS["terminal_growth_rate"], "terminal_growth_rate"
        ),
        "exit_ebitda_multiple": peer_median_ebitda,
    }
    return _validate_assumptions(raw_assumptions)


def _extract_bridge_inputs(target_row: pd.Series) -> Dict[str, float]:
    """Extract and validate EV-to-equity bridge inputs."""
    shares = _required_row_value(target_row, "shares_outstanding", "target_row")
    if shares <= 0:
        raise ValueError("shares_outstanding must be positive")

    return {
        "cash": _optional_row_value(target_row, "cash", "target_row"),
        "debt": _optional_row_value(target_row, "total_debt", "target_row"),
        "minority_interest": _optional_row_value(
            target_row, "minority_interest", "target_row"
        ),
        "shares": shares,
    }


def _bridge_ev_to_equity(
    enterprise_value: float,
    bridge_inputs: Dict[str, float],
) -> Dict[str, float]:
    """Bridge enterprise value to implied equity value and share price."""
    cash = bridge_inputs["cash"]
    debt = bridge_inputs["debt"]
    minority_interest = bridge_inputs["minority_interest"]
    shares = bridge_inputs["shares"]

    equity_value = enterprise_value + cash - debt - minority_interest
    return {
        "enterprise_value": enterprise_value,
        "plus_cash": cash,
        "less_debt": debt,
        "less_minority_interest": minority_interest,
        "implied_equity_value": equity_value,
        "diluted_shares": shares,
        "implied_share_price": equity_value / shares,
    }


def _project_ufcf_schedule(
    base_year: int,
    initial_revenue: float,
    assumptions: Dict[str, Any],
    wacc: float,
) -> pd.DataFrame:
    """Build the annual unlevered free cash flow projections."""
    projections = []
    previous_revenue = initial_revenue

    for period in range(1, assumptions["projection_years"] + 1):
        revenue = previous_revenue * (1.0 + assumptions["revenue_growth"])
        delta_revenue = revenue - previous_revenue

        ebitda = revenue * assumptions["ebitda_margin"]
        dna = revenue * assumptions["dna_pct_rev"]
        ebit = ebitda - dna
        nopat = ebit * (1.0 - assumptions["tax_rate"])
        capex = revenue * assumptions["capex_pct_rev"]
        delta_nwc = delta_revenue * assumptions["nwc_pct_delta_rev"]
        ufcf = nopat + dna - capex - delta_nwc

        discount_factor = 1.0 / ((1.0 + wacc) ** period)
        projections.append(
            {
                "period": period,
                "fiscal_year": f"{base_year + period}E",
                "revenue": revenue,
                "ebitda": ebitda,
                "dna": dna,
                "ebit": ebit,
                "nopat": nopat,
                "capex": capex,
                "delta_nwc": delta_nwc,
                "ufcf": ufcf,
                "discount_factor": discount_factor,
                "pv_ufcf": ufcf * discount_factor,
            }
        )
        previous_revenue = revenue

    return pd.DataFrame(projections)


def _calculate_terminal_values(
    proj_df: pd.DataFrame,
    assumptions: Dict[str, Any],
    wacc: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate terminal values using the perpetuity-growth and exit-multiple methods."""
    terminal_ufcf = _finite_float(proj_df["ufcf"].iloc[-1], "terminal_ufcf")
    terminal_ebitda = _finite_float(proj_df["ebitda"].iloc[-1], "terminal_ebitda")
    terminal_discount = _finite_float(
        proj_df["discount_factor"].iloc[-1], "terminal_discount_factor"
    )
    terminal_growth = assumptions["terminal_growth_rate"]

    if terminal_ufcf <= 0:
        logger.warning(
            "Terminal UFCF is non-positive (%s); perpetuity valuation may be "
            "economically unmeaningful",
            terminal_ufcf,
        )

    # A negative EBITDA implies no positive EBITDA-based operating value; do not
    # apply a positive trading multiple to an unprofitable terminal year.
    if terminal_ebitda <= 0:
        logger.warning(
            "Terminal EBITDA is non-positive (%s); exit-multiple valuation is "
            "reported as unavailable",
            terminal_ebitda,
        )
        exit_value = np.nan
        pv_exit_value = np.nan
    else:
        exit_value = terminal_ebitda * assumptions["exit_ebitda_multiple"]
        pv_exit_value = exit_value * terminal_discount

    perpetuity_value = (
        terminal_ufcf * (1.0 + terminal_growth) / (wacc - terminal_growth)
    )
    pv_perpetuity_value = perpetuity_value * terminal_discount

    return (
        {
            "terminal_value": perpetuity_value,
            "pv_terminal_value": pv_perpetuity_value,
        },
        {
            "terminal_value": exit_value,
            "pv_terminal_value": pv_exit_value,
        },
    )


def run_dcf_model(
    hist_df: pd.DataFrame,
    comps_df: pd.DataFrame,
    custom_assumptions: Optional[Dict[str, Any]] = None,
    wacc_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a forecast DCF and bridge enterprise value to implied equity value."""
    if hist_df.empty:
        raise ValueError("hist_df is empty")

    target_row = _get_target_row(comps_df)
    latest_hist = hist_df.iloc[-1]
    wacc_data = calculate_wacc(target_row, latest_hist)

    wacc = (
        wacc_data["wacc"]
        if wacc_override is None
        else _finite_float(wacc_override, "wacc_override")
    )
    if wacc <= 0:
        raise ValueError("WACC must be positive")

    assumptions = derive_dcf_assumptions(hist_df, comps_df)
    if custom_assumptions:
        unknown = set(custom_assumptions).difference(assumptions)
        if unknown:
            raise ValueError(f"Unknown custom assumptions: {sorted(unknown)}")
        assumptions = _validate_assumptions(
            {**assumptions, **custom_assumptions}
        )

    terminal_growth = assumptions["terminal_growth_rate"]
    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC ({wacc:.4%}) must exceed terminal growth "
            f"({terminal_growth:.4%})"
        )

    if "fiscal_year" not in latest_hist.index:
        raise ValueError("Latest historical row is missing 'fiscal_year'")
    try:
        base_year = int(str(latest_hist["fiscal_year"])[:4])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid latest fiscal year: {latest_hist['fiscal_year']!r}"
        ) from exc

    initial_revenue = _required_row_value(latest_hist, "revenue", "latest_hist")
    if initial_revenue <= 0:
        raise ValueError("Latest historical revenue must be positive")

    projections = _project_ufcf_schedule(
        base_year, initial_revenue, assumptions, wacc
    )
    sum_pv_ufcf = _finite_float(projections["pv_ufcf"].sum(), "sum_pv_ufcf")
    perpetuity_values, exit_values = _calculate_terminal_values(
        projections, assumptions, wacc
    )
    bridge_inputs = _extract_bridge_inputs(target_row)

    perpetuity_result = {
        **perpetuity_values,
        **_bridge_ev_to_equity(
            sum_pv_ufcf + perpetuity_values["pv_terminal_value"],
            bridge_inputs,
        ),
    }

    if pd.isna(exit_values["pv_terminal_value"]):
        exit_result = {
            **exit_values,
            "enterprise_value": np.nan,
            "plus_cash": bridge_inputs["cash"],
            "less_debt": bridge_inputs["debt"],
            "less_minority_interest": bridge_inputs["minority_interest"],
            "implied_equity_value": np.nan,
            "diluted_shares": bridge_inputs["shares"],
            "implied_share_price": np.nan,
        }
    else:
        exit_result = {
            **exit_values,
            **_bridge_ev_to_equity(
                sum_pv_ufcf + exit_values["pv_terminal_value"],
                bridge_inputs,
            ),
        }

    return {
        "wacc_schedule": {
            **wacc_data,
            "wacc": wacc,
            "wacc_override": wacc_override,
        },
        "assumptions": assumptions,
        "projections": projections,
        "sum_pv_ufcf": sum_pv_ufcf,
        "terminal_perpetuity": perpetuity_result,
        "terminal_exit_multiple": exit_result,
    }


def _peer_quantiles(
    peers_df: pd.DataFrame,
    column: str,
) -> Tuple[float, float, float]:
    """Compute 25th, 50th, and 75th percentiles for positive peer multiples."""
    if column not in peers_df.columns:
        raise ValueError(f"comps_df is missing peer multiple column {column!r}")

    values = (
        pd.to_numeric(peers_df[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    values = values[values > 0]
    if values.empty:
        raise ValueError(f"No valid positive peer multiples for {column}")

    quantiles = values.quantile([0.25, 0.50, 0.75])
    return tuple(float(quantiles.loc[q]) for q in (0.25, 0.50, 0.75))


def _dcf_wacc_sensitivity_bounds(
    hist_df: pd.DataFrame,
    comps_df: pd.DataFrame,
    dcf_results: Dict[str, Any],
    base_wacc: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Recalculate DCF prices at WACC +/- 50 bps.

    If a sensitivity rate does not exceed terminal growth, the base case is used
    for that endpoint. Returned ranges are sorted by implied price.
    """
    base_wacc = _finite_float(base_wacc, "base_wacc")
    if base_wacc <= 0:
        raise ValueError("base_wacc must be positive")

    assumptions = dict(dcf_results["assumptions"])
    growth = _finite_float(
        assumptions["terminal_growth_rate"], "terminal_growth_rate"
    )

    def sensitivity_case(rate: float, label: str) -> Dict[str, Any]:
        if rate <= growth or rate <= 0:
            logger.warning(
                "%s sensitivity WACC %.4%% is not above terminal growth %.4%%; "
                "using base case",
                label,
                rate * 100,
                growth * 100,
            )
            return dcf_results

        return run_dcf_model(
            hist_df,
            comps_df,
            custom_assumptions=assumptions,
            wacc_override=rate,
        )

    high_rate_case = sensitivity_case(base_wacc + 0.005, "High-WACC")
    low_rate_case = sensitivity_case(base_wacc - 0.005, "Low-WACC")

    def price(result: Dict[str, Any], method: str) -> float:
        value = result[method]["implied_share_price"]
        return _finite_float(value, f"{method}.implied_share_price")

    perpetuity_bounds = tuple(
        sorted(
            (
                price(high_rate_case, "terminal_perpetuity"),
                price(low_rate_case, "terminal_perpetuity"),
            )
        )
    )
    exit_prices = [
        high_rate_case["terminal_exit_multiple"]["implied_share_price"],
        low_rate_case["terminal_exit_multiple"]["implied_share_price"],
    ]
    valid_exit_prices = [
        _finite_float(value, "terminal_exit_multiple.implied_share_price")
        for value in exit_prices
        if not _is_missing_scalar(value)
    ]

    # No exit-multiple range is available if terminal EBITDA is non-positive.
    if len(valid_exit_prices) != 2:
        exit_bounds = (np.nan, np.nan)
    else:
        exit_bounds = tuple(sorted(valid_exit_prices))

    return perpetuity_bounds, exit_bounds


def build_football_field(
    comps_df: pd.DataFrame,
    dcf_results: Dict[str, Any],
    hist_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build DCF and trading-comps share-price ranges."""
    target_row = _get_target_row(comps_df)
    target_mask = comps_df["is_target"].map(_target_flag)
    peers_only = comps_df[~target_mask]

    target_ebitda = _required_row_value(target_row, "ltm_ebitda", "target_row")
    target_revenue = _required_row_value(target_row, "ltm_revenue", "target_row")
    current_price = _required_row_value(target_row, "share_price", "target_row")
    if target_ebitda <= 0 or target_revenue <= 0:
        raise ValueError("Target LTM EBITDA and revenue must be positive")

    bridge_inputs = _extract_bridge_inputs(target_row)
    ebitda_q25, ebitda_median, ebitda_q75 = _peer_quantiles(
        peers_only, "ev_to_ebitda"
    )
    revenue_q25, revenue_median, revenue_q75 = _peer_quantiles(
        peers_only, "ev_to_revenue"
    )

    perpetuity_mid = _finite_float(
        dcf_results["terminal_perpetuity"]["implied_share_price"],
        "terminal_perpetuity.implied_share_price",
    )
    exit_mid_raw = dcf_results["terminal_exit_multiple"]["implied_share_price"]
    exit_mid = (
        np.nan
        if _is_missing_scalar(exit_mid_raw)
        else _finite_float(exit_mid_raw, "terminal_exit_multiple.implied_share_price")
    )
    base_wacc = _finite_float(
        dcf_results["wacc_schedule"]["wacc"], "wacc_schedule.wacc"
    )

    if hist_df is not None and not hist_df.empty:
        (perp_low, perp_high), (exit_low, exit_high) = (
            _dcf_wacc_sensitivity_bounds(
                hist_df, comps_df, dcf_results, base_wacc
            )
        )
    else:
        perp_low, perp_high = sorted(
            [perpetuity_mid * 0.88, perpetuity_mid * 1.15]
        )
        if _is_missing_scalar(exit_mid):
            exit_low, exit_high = np.nan, np.nan
        else:
            exit_low, exit_high = sorted([exit_mid * 0.90, exit_mid * 1.12])

    def to_price(enterprise_value: float) -> float:
        return _bridge_ev_to_equity(enterprise_value, bridge_inputs)[
            "implied_share_price"
        ]

    rows = [
        {
            "methodology": "DCF (Perpetuity Growth)",
            "low_price": perp_low,
            "mid_price": perpetuity_mid,
            "high_price": perp_high,
        },
        {
            "methodology": "DCF (Exit EV/EBITDA Multiple)",
            "low_price": exit_low,
            "mid_price": exit_mid,
            "high_price": exit_high,
        },
        {
            "methodology": "Trading Comps (EV / LTM EBITDA)",
            "low_price": to_price(target_ebitda * ebitda_q25),
            "mid_price": to_price(target_ebitda * ebitda_median),
            "high_price": to_price(target_ebitda * ebitda_q75),
        },
        {
            "methodology": "Trading Comps (EV / LTM Revenue)",
            "low_price": to_price(target_revenue * revenue_q25),
            "mid_price": to_price(target_revenue * revenue_median),
            "high_price": to_price(target_revenue * revenue_q75),
        },
    ]

    football_field = pd.DataFrame(rows)
    football_field["current_price"] = current_price
    football_field["base_wacc"] = base_wacc
    return football_field


if __name__ == "__main__":
    from src.fetcher import fetch_all_companies
    from src.normalizer import build_comps_table, normalize_company_historicals

    raw_data = fetch_all_companies(force_refresh=False)
    target_ticker = str(TARGET_TICKER).strip().upper()
    normalized_payloads = {
        str(ticker).strip().upper(): payload
        for ticker, payload in raw_data.items()
    }

    if target_ticker not in normalized_payloads:
        raise SystemExit(f"Target ticker {target_ticker} was not fetched")

    target_history = normalize_company_historicals(
        normalized_payloads[target_ticker]
    )
    comps = build_comps_table(normalized_payloads)
    dcf_output = run_dcf_model(target_history, comps)
    football_field = build_football_field(
        comps,
        dcf_output,
        hist_df=target_history,
    )

    print(f"\nComputed WACC: {dcf_output['wacc_schedule']['wacc']:.2%}")
    print("\n--- Projected UFCF ---")
    print(
        dcf_output["projections"][
            ["fiscal_year", "revenue", "ebitda", "ufcf", "pv_ufcf"]
        ]
    )
    print("\n--- Valuation Summary ---")
    print(
        football_field[
            [
                "methodology",
                "low_price",
                "mid_price",
                "high_price",
                "current_price",
            ]
        ]
    )
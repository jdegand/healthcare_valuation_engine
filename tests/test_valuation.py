import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import TARGET_TICKER
from src.excel_builder import build_valuation_workbook
from src.normalizer import build_comps_table, normalize_company_historicals
from src.valuation import (
    build_football_field,
    calculate_wacc,
    derive_dcf_assumptions,
    run_dcf_model,
)


@pytest.fixture
def sample_payloads() -> dict:
    """Synthetic but realistic multi-year payloads for target and two peers."""
    target = str(TARGET_TICKER).strip().upper()
    return {
        target: {
            "ticker": target,
            "market_snapshot": {
                "ticker": target,
                "company_name": "Target Hospital Corp",
                "current_price": 100.0,
                "market_cap": 10_000.0,
                "enterprise_value": 14_500.0,
                "shares_outstanding": 100.0,
                "beta": 1.10,
                "trailing_pe": 14.0,
                "forward_pe": 12.5,
                "total_debt": 5_000.0,
                "total_cash": 1_000.0,
            },
            "income_statement": {
                # Null stub year that must be filtered out automatically
                "2022-12-31": {"Total Revenue": None, "EBITDA": None},
                "2023-12-31": {
                    "Total Revenue": 8_000.0,
                    "EBITDA": 1_600.0,
                    "EBIT": 1_200.0,
                    "Pretax Income": 1_000.0,
                    "Tax Provision": 210.0,
                },
                "2024-12-31": {
                    "Total Revenue": 8_800.0,
                    "EBITDA": 1_760.0,
                    "EBIT": 1_320.0,
                    "Pretax Income": 1_100.0,
                    "Tax Provision": 231.0,
                },
            },
            "balance_sheet": {
                "2023-12-31": {
                    "Current Assets": 2_000.0,
                    "Cash And Cash Equivalents": 800.0,
                    "Current Liabilities": 1_000.0,
                    "Current Debt": 200.0,
                    "Total Debt": 5_000.0,
                    "Minority Interest": 500.0,
                },
                "2024-12-31": {
                    "Current Assets": 2_200.0,
                    "Cash And Cash Equivalents": 1_000.0,
                    "Current Liabilities": 1_050.0,
                    "Current Debt": 250.0,
                    "Total Debt": 5_000.0,
                    "Minority Interest": 500.0,
                },
            },
            "cash_flow": {
                "2023-12-31": {
                    "Depreciation And Amortization": 400.0,
                    "Capital Expenditure": -480.0,  # Negative outflow in source
                },
                "2024-12-31": {
                    "Depreciation And Amortization": 440.0,
                    "Capital Expenditure": -528.0,
                },
            },
        },
        "PEER1": {
            "ticker": "PEER1",
            "market_snapshot": {
                "ticker": "PEER1",
                "company_name": "Peer One Health",
                "current_price": 50.0,
                "market_cap": 5_000.0,
                "shares_outstanding": 100.0,
                "beta": 1.20,
                "trailing_pe": -5.0,  # Negative P/E should become NaN
                "forward_pe": -3.2,   # Negative P/E should become NaN
                "total_debt": 2_000.0,
                "total_cash": 500.0,
            },
            "income_statement": {
                "2024-12-31": {
                    "Total Revenue": 5_000.0,
                    "EBITDA": 1_000.0,
                    "EBIT": 750.0,
                    "Pretax Income": 600.0,
                    "Tax Provision": 150.0,
                }
            },
            "balance_sheet": {
                "2024-12-31": {
                    "Current Assets": 1_000.0,
                    "Cash And Cash Equivalents": 500.0,
                    "Current Liabilities": 600.0,
                    "Current Debt": 100.0,
                    "Total Debt": 2_000.0,
                    # Intentionally omit Minority Interest to test NaN handling
                }
            },
            "cash_flow": {
                "2024-12-31": {
                    "Depreciation And Amortization": 250.0,
                    "Capital Expenditure": -300.0,
                }
            },
        },
        "PEER2": {
            "ticker": "PEER2",
            "market_snapshot": {
                "ticker": "PEER2",
                "company_name": "Peer Two Medical",
                "current_price": 80.0,
                "market_cap": 8_000.0,
                "shares_outstanding": 100.0,
                "beta": 0.95,
                "trailing_pe": 12.0,
                "forward_pe": 11.0,
                "total_debt": 2_000.0,
                "total_cash": 1_000.0,
            },
            "income_statement": {
                "2024-12-31": {
                    "Total Revenue": 6_000.0,
                    "EBITDA": 1_500.0,
                    "EBIT": 1_200.0,
                    "Pretax Income": 1_000.0,
                    "Tax Provision": 200.0,
                }
            },
            "balance_sheet": {
                "2024-12-31": {
                    "Current Assets": 1_200.0,
                    "Cash And Cash Equivalents": 1_000.0,
                    "Current Liabilities": 700.0,
                    "Current Debt": 200.0,
                    "Total Debt": 2_000.0,
                    "Minority Interest": 1_000.0,
                }
            },
            "cash_flow": {
                "2024-12-31": {
                    "Depreciation And Amortization": 300.0,
                    "Capital Expenditure": 360.0,  # Already positive in source
                }
            },
        },
    }


def test_normalizer_drops_null_stubs_and_standardizes_ufcf(sample_payloads):
    """Verify null stub years are dropped, CapEx is positive, and UFCF matches manual math."""
    target = str(TARGET_TICKER).strip().upper()
    df = normalize_company_historicals(sample_payloads[target])

    # 2022 null stub must be removed; only 2023 and 2024 remain
    assert list(df["fiscal_year"]) == ["2023", "2024"]

    # CapEx must be positive regardless of source sign
    assert df.loc[0, "capex"] == pytest.approx(480.0)
    assert df.loc[1, "capex"] == pytest.approx(528.0)

    # 2023 NWC = (2000 - 800) - (1000 - 200) = 400; first year delta_nwc = 0
    # 2024 NWC = (2200 - 1000) - (1050 - 250) = 400; delta_nwc = 0
    assert df.loc[0, "nwc"] == pytest.approx(400.0)
    assert df.loc[1, "delta_nwc"] == pytest.approx(0.0)

    # 2024 NOPAT = 1320 * (1 - 0.21) = 1042.8; UFCF = 1042.8 + 440 - 528 - 0 = 954.8
    assert df.loc[1, "ufcf"] == pytest.approx(954.8)


def test_normalizer_incomplete_balance_sheet_produces_nan_not_zero(sample_payloads):
    """Missing Current Liabilities should produce NaN NWC/UFCF rather than silent zero."""
    payload = sample_payloads["PEER1"]
    del payload["balance_sheet"]["2024-12-31"]["Current Liabilities"]

    df = normalize_company_historicals(payload)
    assert pd.isna(df.loc[0, "nwc"])
    assert pd.isna(df.loc[0, "ufcf"])


def test_comps_table_includes_minority_interest_and_filters_negative_pe(sample_payloads):
    """EV must include Minority Interest, and negative P/E ratios must become NaN."""
    comps = build_comps_table(sample_payloads)
    target = str(TARGET_TICKER).strip().upper()

    hca_row = comps[comps["ticker"] == target].iloc[0]
    # EV = 10,000 (Mkt Cap) + 5,000 (Debt) + 500 (Minority Int) - 1,000 (Cash) = 14,500
    assert hca_row["enterprise_value"] == pytest.approx(14_500.0)
    assert hca_row["ev_to_ebitda"] == pytest.approx(14_500.0 / 1_760.0)

    peer1_row = comps[comps["ticker"] == "PEER1"].iloc[0]
    assert pd.isna(peer1_row["trailing_pe"])
    assert pd.isna(peer1_row["forward_pe"])
    # Missing Minority Interest in PEER1 should not break EV (5000 + 2000 - 500 = 6500)
    assert peer1_row["enterprise_value"] == pytest.approx(6_500.0)

def test_wacc_and_ev_to_equity_bridge_math(sample_payloads):
    """Verify CAPM WACC, Gordon Growth Terminal Value, and Equity Value bridge."""
    target = str(TARGET_TICKER).strip().upper()
    hist_df = normalize_company_historicals(sample_payloads[target])
    comps_df = build_comps_table(sample_payloads)

    results = run_dcf_model(
        hist_df,
        comps_df,
        custom_assumptions={
            "projection_years": 5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.20,
            "dna_pct_rev": 0.05,
            "capex_pct_rev": 0.06,
            "nwc_pct_delta_rev": 0.01,
            "tax_rate": 0.21,
            "terminal_growth_rate": 0.025,
            "exit_ebitda_multiple": 7.0,
        },
        wacc_override=0.08,
    )

    perp = results["terminal_perpetuity"]
    # Equity Value = EV + Cash (1000) - Debt (5000) - Minority Interest (500) = EV - 4500
    assert perp["implied_equity_value"] == pytest.approx(perp["enterprise_value"] - 4_500.0)
    assert perp["implied_share_price"] == pytest.approx(perp["implied_equity_value"] / 100.0)


def test_edge_case_wacc_below_or_equal_terminal_growth_raises(sample_payloads):
    """Gordon Growth formula must raise ValueError when WACC <= terminal_growth_rate."""
    target = str(TARGET_TICKER).strip().upper()
    hist_df = normalize_company_historicals(sample_payloads[target])
    comps_df = build_comps_table(sample_payloads)

    with pytest.raises(ValueError, match="must exceed terminal growth"):
        run_dcf_model(
            hist_df,
            comps_df,
            custom_assumptions={"terminal_growth_rate": 0.08},
            wacc_override=0.08,
        )


def test_edge_case_pd_na_in_is_target_and_ticker_mismatch(sample_payloads):
    """_get_target_row must handle pd.NA in is_target and reject mismatched target tickers."""
    target = str(TARGET_TICKER).strip().upper()
    hist_df = normalize_company_historicals(sample_payloads[target])
    comps_df = build_comps_table(sample_payloads).astype({"is_target": "object"})

    # Setting a peer's is_target to pd.NA must not raise ambiguous truth value error
    comps_df.loc[comps_df["ticker"] == "PEER1", "is_target"] = pd.NA
    out = run_dcf_model(hist_df, comps_df)
    assert math.isfinite(out["terminal_perpetuity"]["implied_share_price"])

    # Flagging a non-target ticker as is_target=True must raise descriptive ValueError
    bad_comps = comps_df.copy()
    bad_comps["is_target"] = bad_comps["ticker"] == "PEER1"
    with pytest.raises(ValueError, match="do not match TARGET_TICKER"):
        run_dcf_model(hist_df, bad_comps)


def test_edge_case_negative_terminal_ebitda_yields_nan_exit_multiple(sample_payloads):
    """Negative terminal EBITDA should mark exit-multiple valuation as NaN instead of negative EV."""
    target = str(TARGET_TICKER).strip().upper()
    hist_df = normalize_company_historicals(sample_payloads[target])
    comps_df = build_comps_table(sample_payloads)

    results = run_dcf_model(
        hist_df,
        comps_df,
        custom_assumptions={"ebitda_margin": -0.05},
    )
    assert pd.isna(results["terminal_exit_multiple"]["terminal_value"])
    assert pd.isna(results["terminal_exit_multiple"]["implied_share_price"])

    ff = build_football_field(comps_df, results, hist_df=hist_df)
    exit_row = ff[ff["methodology"] == "DCF (Exit EV/EBITDA Multiple)"].iloc[0]
    assert pd.isna(exit_row["low_price"])
    assert pd.isna(exit_row["high_price"])


def test_edge_case_single_year_history_and_zero_debt(sample_payloads):
    """Single historical year and debt-free target should fall back cleanly without crashing."""
    target = str(TARGET_TICKER).strip().upper()
    single_year_payload = sample_payloads[target].copy()
    single_year_payload["income_statement"] = {
        "2024-12-31": sample_payloads[target]["income_statement"]["2024-12-31"]
    }
    hist_df = normalize_company_historicals(single_year_payload)
    comps_df = build_comps_table(sample_payloads)

    # Zero debt & missing minority interest on target
    comps_df.loc[comps_df["ticker"] == target, "total_debt"] = 0.0
    comps_df.loc[comps_df["ticker"] == target, "minority_interest"] = np.nan

    assumptions = derive_dcf_assumptions(hist_df, comps_df)
    assert assumptions["revenue_growth"] == pytest.approx(0.05)

    wacc_info = calculate_wacc(comps_df[comps_df["is_target"]].iloc[0], hist_df.iloc[-1])
    assert wacc_info["weight_equity"] == pytest.approx(1.0)
    assert wacc_info["weight_debt"] == pytest.approx(0.0)


def test_excel_builder_end_to_end_with_reordered_comps(sample_payloads, tmp_path, monkeypatch):
    """Excel builder must reorder target to row 4 and write valid workbook even with NaNs."""
    import src.excel_builder as eb

    monkeypatch.setattr(eb, "OUTPUT_DIR", tmp_path)

    target = str(TARGET_TICKER).strip().upper()
    hist_df = normalize_company_historicals(sample_payloads[target])
    # Reverse comps order so target is NOT first in the DataFrame
    comps_df = build_comps_table(sample_payloads).iloc[::-1].reset_index(drop=True)
    dcf_results = run_dcf_model(hist_df, comps_df)
    ff_df = build_football_field(comps_df, dcf_results, hist_df=hist_df)

    out_path = build_valuation_workbook(
        hist_df=hist_df,
        comps_df=comps_df,
        dcf_results=dcf_results,
        football_field_df=ff_df,
        output_filename="test_model.xlsx",
    )
    assert Path(out_path).exists()

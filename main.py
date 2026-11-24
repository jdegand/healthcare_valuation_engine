import argparse
import logging
import sys

from config import TARGET_TICKER
from src.excel_builder import build_valuation_workbook
from src.fetcher import fetch_all_companies
from src.normalizer import build_comps_table, normalize_company_historicals
from src.valuation import build_football_field, run_dcf_model

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for the CLI pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Healthcare Provider DCF & Trading Comps Valuation Engine"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh financial statements & quotes from Yahoo Finance instead of using local cache",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output .xlsx filename (defaults to <TARGET_TICKER>_Valuation_Model.xlsx)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    target_ticker = str(TARGET_TICKER).strip().upper()

    # 1. Fetch raw financial payloads (src/fetcher.py)
    logger.info("Step 1/4: Fetching financial statements & market snapshots...")
    raw_payloads = fetch_all_companies(force_refresh=args.refresh)
    normalized_payloads = {
        str(sym).strip().upper(): payload for sym, payload in raw_payloads.items()
    }

    if target_ticker not in normalized_payloads:
        logger.error("Target ticker %s could not be retrieved. Aborting.", target_ticker)
        return 1

    # 2. Normalize historicals & build Trading Comps table (src/normalizer.py)
    logger.info("Step 2/4: Normalizing historical financials & building Trading Comps...")
    target_hist = normalize_company_historicals(normalized_payloads[target_ticker])
    comps_table = build_comps_table(normalized_payloads)

    # 3. Run DCF valuation & Football Field ranges (src/valuation.py)
    logger.info("Step 3/4: Running DCF projection, WACC, & Football Field valuation...")
    dcf_results = run_dcf_model(target_hist, comps_table)
    football_field = build_football_field(
        comps_df=comps_table,
        dcf_results=dcf_results,
        hist_df=target_hist,
    )

    # 4. Generate dynamic multi-tab Excel workbook (src/excel_builder.py)
    logger.info("Step 4/4: Exporting formatted Excel valuation model...")
    output_filename = args.output or f"{target_ticker}_Valuation_Model.xlsx"
    output_path = build_valuation_workbook(
        hist_df=target_hist,
        comps_df=comps_table,
        dcf_results=dcf_results,
        football_field_df=football_field,
        output_filename=output_filename,
    )

    # Print quick terminal summary
    wacc = dcf_results["wacc_schedule"]["wacc"]
    perp_price = dcf_results["terminal_perpetuity"]["implied_share_price"]
    exit_price = dcf_results["terminal_exit_multiple"]["implied_share_price"]
    curr_price = float(football_field["current_price"].iloc[0])

    print("\n============================================================")
    print(f"  {target_ticker} Valuation Summary  (WACC: {wacc:.2%})")
    print("============================================================")
    print(f"  Current Share Price:             ${curr_price:,.2f}")
    print(f"  DCF (Perpetuity Growth):         ${perp_price:,.2f}")
    print(f"  DCF (Exit EV/EBITDA Multiple):   ${exit_price:,.2f}")
    print("------------------------------------------------------------")
    print(f"  Excel Workbook Saved To: {output_path}")
    print("============================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

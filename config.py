from pathlib import Path

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target & Institutional Hospital / Outpatient Peer Group (Excludes distressed CYH)
TARGET_TICKER = "HCA"
PEER_TICKERS = ["THC", "UHS", "ARDT", "EHC", "SGRY"]
ALL_TICKERS = [TARGET_TICKER] + PEER_TICKERS

# Caching Settings
CACHE_EXPIRY_HOURS = 24  # Re-fetch from Yahoo Finance if cached file is older than this

# Default DCF & WACC Assumptions (Baseline)
DCF_DEFAULTS = {
    "projection_years": 5,
    "risk_free_rate": 0.0425,        # ~4.25% 10Y Treasury baseline
    "equity_risk_premium": 0.0500,   # 5.00% ERP
    "terminal_growth_rate": 0.0250,  # 2.50% perpetuity growth
    "default_tax_rate": 0.2300,      # 23.0% effective corporate tax rate fallback
    "cost_of_debt_pretax": 0.0575,   # 5.75% pre-tax borrowing cost baseline
    "default_exit_multiple": 9.0,    # Anchored to HCA's historical ~9.0x quality EV/EBITDA multiple
}

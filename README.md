# Automated Healthcare Provider Valuation & Trading Comps Engine

A Python-based corporate finance pipeline that ingests live market quotes and historical 10-K financial statements for **HCA Healthcare (`HCA`)** and its publicly traded hospital peer group (**`THC`**, **`UHS`**, **`CYH`**), normalizes accounting line items, runs an Unlevered Discounted Cash Flow (DCF) and Trading Comps valuation, and exports an institutional-grade, formula-driven 4-tab Excel model.

---

## Key Features

* **Fault-Tolerant Ingestion & Atomic Caching (`src/fetcher.py`):** Pulls annual Income Statements, Balance Sheets, Cash Flow Statements, and live market snapshots via `yfinance` with 24-hour local JSON caching (`data/raw/`) and atomic `.tmp` file replacement.
* **Healthcare Accounting Normalization (`src/normalizer.py`):**
  * Filters out incomplete fiscal-year stub columns before computing year-over-year changes.
  * Standardizes Capital Expenditures as positive cash outflows for Unlevered Free Cash Flow (\(\text{NOPAT} + \text{D&A} - \text{CapEx} - \Delta\text{NWC}\)).
  * Incorporates **Noncontrolling (Minority) Interest** into Enterprise Value (Market Cap + Total Debt + Minority Interest - Cash) so EV aligns with consolidated hospital joint-venture EBITDA.
  * Preserves missing balance-sheet inputs as `NaN` rather than defaulting to zero, and excludes negative P/E ratios from peer summary statistics.
* **DCF, WACC & Sensitivity Engine (`src/valuation.py`):**
  * Computes WACC via CAPM using live Beta, market-value capital weights, and effective tax rates.
  * Projects 5-year Unlevered Free Cash Flows and calculates Terminal Value under both the **Perpetuity Growth (Gordon Growth)** and **Peer Median EV/EBITDA Exit Multiple** methods.
  * Recalculates DCF valuation bounds at WACC ± 50 bps for the Football Field summary.
* **Dynamic Multi-Tab Excel Report (`src/excel_builder.py`):** Generates `output/HCA_Valuation_Model.xlsx` with live Excel formulas (`=NPV`, `=SUM`, `=MEDIAN`, `=PERCENTILE`), Wall Street color coding (blue inputs, black same-sheet formulas, green cross-sheet links), a **Bull / Base / Bear scenario dropdown (`INDEX/MATCH`)**, two 2-variable sensitivity matrices, and a floating-bar **Football Field** chart.

---

## Workbook Architecture (`HCA_Valuation_Model.xlsx`)

| Tab | Contents |
| :--- | :--- |
| **1. Executive Summary** | Valuation Football Field summary table (`Low`, `Base/Mid`, `High`, `Current Price`), floating horizontal range chart, and methodology notes. |
| **2. DCF Model** | CAPM WACC schedule, interactive **Bull / Base / Bear** scenario selector (`H5`), historical & 5-year projected UFCF schedule, and Enterprise-to-Equity bridge. |
| **3. Sensitivity Analysis** | Dynamic Excel formula tables testing Implied Share Price across **WACC vs. Terminal Growth Rate** and **WACC vs. Exit EV/EBITDA Multiple**. |
| **4. Trading Comps** | Live valuation multiples (`EV/Revenue`, `EV/EBITDA`, `Forward P/E`) across `HCA`, `THC`, `UHS`, and `CYH` with `25th Percentile`, `Median`, `Average`, and `75th Percentile` benchmarks. |

---

## Repository Structure

```text
healthcare-valuation-engine/
├── .gitignore                  # Ignores venv, caches, and dynamic output runs
├── pytest.ini                  # Pytest root path configuration
├── README.md                   # Project documentation & financial methodology
├── requirements.txt            # Pinned Python dependencies
├── config.py                   # Target ticker ('HCA'), peer list, & WACC defaults
├── main.py                     # CLI entry point orchestrating the end-to-end pipeline
│
├── src/
│   ├── __init__.py
│   ├── fetcher.py              # Yahoo Finance data ingestion & atomic JSON cache
│   ├── normalizer.py           # Accounting cleanup, sign standardization, & ratios
│   ├── valuation.py            # WACC, 5Y DCF, EV-to-Equity bridge, & Football Field
│   └── excel_builder.py        # Formatted .xlsx report generator via openpyxl
│
├── tests/
│   ├── __init__.py
│   └── test_valuation.py       # Offline unit & integration tests (pytest)
│
├── data/
│   └── raw/                    # Local JSON cache (*_raw.json)
│
└── output/
    └── HCA_Valuation_Model.xlsx  # Generated dynamic Excel valuation workbook
```

---

## Quickstart

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Valuation Pipeline

```bash
# Run using local cache (or fetch if cache is missing/expired)
python3 main.py

# Force a live refresh from Yahoo Finance
python3 main.py --refresh

# Export to a custom filename
python3 main.py --output HCA_Valuation_Model_Sample.xlsx
```

### 3. Run the Unit Test Suite

All tests run offline using synthetic multi-year financial fixtures:

```bash
pytest -v
```

---

## Financial Methodology & Sector Notes

* **Minority Interest Alignment:** Acute-care hospital operators frequently operate outpatient surgery centers and hospitals through majority-owned joint ventures. Because consolidated EBITDA and UFCF reflect 100% of joint-venture operations, noncontrolling interests are added to Enterprise Value in Trading Comps and subtracted in the DCF Enterprise-to-Equity bridge.
* **WACC Capital Weights:** Per standard corporate finance convention, WACC weights use market value of common equity and total debt, excluding book-value minority interest where standalone market values and betas are unobservable.
* **Peer Multiple Dispersion:** `HCA` commands a valuation premium (~9.9x EV/EBITDA) relative to smaller or more leveraged hospital peers (`THC`, `UHS`, `CYH`, trading around 5.4x–8.1x EV/EBITDA). Consequently, peer-multiple-based methods (`Exit EV/EBITDA` and `Trading Comps`) imply lower standalone share prices than `HCA`'s intrinsic Perpetuity Growth DCF.

## Conclusion

HCA appears roughly fairly valued on an intrinsic DCF basis (~$497/sh vs. $476/sh market price), meaning the stock's forward return depends heavily on whether HCA can sustain its ~20% EBITDA margins and defend its ~9.9x valuation premium over smaller, higher-leverage hospital peers.

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.legend import LegendEntry
from openpyxl.drawing.fill import PatternFillProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from config import OUTPUT_DIR, TARGET_TICKER

logger = logging.getLogger(__name__)

FONT_FAMILY = "Calibri"
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
HIGHLIGHT_FILL = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
TOGGLE_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

FONT_HEADER = Font(name=FONT_FAMILY, size=11, bold=True, color="FFFFFF")
FONT_TITLE = Font(name=FONT_FAMILY, size=14, bold=True, color="1F4E78")
FONT_SUBTITLE = Font(name=FONT_FAMILY, size=9, italic=True, color="595959")
FONT_SECTION = Font(name=FONT_FAMILY, size=11, bold=True, color="1F4E78")
FONT_INPUT = Font(name=FONT_FAMILY, size=10, color="0000FF")
FONT_FORMULA = Font(name=FONT_FAMILY, size=10, color="000000")
FONT_LINK = Font(name=FONT_FAMILY, size=10, color="008000")
FONT_BOLD = Font(name=FONT_FAMILY, size=10, bold=True, color="000000")
FONT_TOGGLE = Font(name=FONT_FAMILY, size=10, bold=True, color="0000FF")
FONT_NOTE = Font(name=FONT_FAMILY, size=9, italic=True, color="404040")

THIN_SIDE = Side(style="thin", color="BFBFBF")
DOUBLE_SIDE = Side(style="double", color="000000")
TOP_THIN_BOTTOM_DOUBLE = Border(top=THIN_SIDE, bottom=DOUBLE_SIDE)
GRID_BORDER = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=THIN_SIDE)

FMT_CURRENCY_M = '$#,##0;($#,##0);"-"'
FMT_PRICE = "$#,##0.00"
FMT_PERCENT = "0.0%"
FMT_WACC = "0.00%"
FMT_MULTIPLE = '0.0"x"'
FMT_INT = "#,##0"


def _excel_num(value: Any, scale: float = 1.0, default: Optional[float] = None) -> Optional[float]:
    """Convert numeric values safely for openpyxl, avoiding float('nan') in cells."""
    if value is None or pd.isna(value):
        return default
    return float(value) / scale


def _autofit_columns(ws, min_width: int = 12, max_width: int = 34) -> None:
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        longest = 0
        for cell in col:
            # Skip wide banner/subtitle cells in rows 1-2
            if cell.row in (1, 2):
                continue
            val = str(cell.value or "")
            if not val.startswith("="):
                longest = max(longest, len(val))
        ws.column_dimensions[col_letter].width = max(min_width, min(longest + 4, max_width))


def _style_header_row(ws, row_idx: int, max_col: int) -> None:
    for c in range(1, max_col + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HEADER_FILL
        cell.font = FONT_HEADER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def build_valuation_workbook(
    hist_df: pd.DataFrame,
    comps_df: pd.DataFrame,
    dcf_results: Dict[str, Any],
    football_field_df: pd.DataFrame,
    output_filename: str = f"{TARGET_TICKER}_Valuation_Model.xlsx",
) -> Path:
    """Build a formatted, formula-driven 4-tab Excel valuation workbook."""
    if comps_df.empty or hist_df.empty:
        raise ValueError("Cannot build Excel workbook with empty comps_df or hist_df")

    ordered_comps = (
        comps_df.assign(_sort_key=~comps_df["is_target"].fillna(False).astype(bool))
        .sort_values("_sort_key")
        .drop(columns=["_sort_key"])
        .reset_index(drop=True)
    )
    target_row = ordered_comps.iloc[0]
    last_actual_fy = str(hist_df["fiscal_year"].iloc[-1])
    last_actual_dt = str(hist_df["fiscal_date"].iloc[-1])
    run_date_str = datetime.now().strftime("%Y-%m-%d")

    wb = Workbook()
    wb.remove(wb.active)

    ws_exec = wb.create_sheet("Executive Summary")
    ws_dcf = wb.create_sheet("DCF Model")
    ws_sens = wb.create_sheet("Sensitivity Analysis")
    ws_comps = wb.create_sheet("Trading Comps")

    for ws in [ws_exec, ws_dcf, ws_sens, ws_comps]:
        ws.views.sheetView[0].showGridLines = True

    # =========================================================================
    # TAB 4: TRADING COMPS
    # =========================================================================
    ws_comps["A1"] = f"{TARGET_TICKER} vs. Acute-Care Hospital Peer Group — Trading Comps"
    ws_comps["A1"].font = FONT_TITLE
    ws_comps["A2"] = (
        f"Market Data As Of: {run_date_str} | LTM Fundamentals As Of: FY{last_actual_fy} ({last_actual_dt}) | "
        f"Units: $ in Millions, except Share Price ($/sh) and Multiples (x)"
    )
    ws_comps["A2"].font = FONT_SUBTITLE

    comp_headers = [
        "Ticker", "Company Name", "Share Price ($)", "Market Cap ($M)", "Total Debt ($M)",
        "Minority Int. ($M)", "Cash ($M)", "Enterprise Value ($M)", "LTM Revenue ($M)", "LTM EBITDA ($M)",
        "EBITDA Margin", "EV / Revenue", "EV / EBITDA", "Forward P/E",
    ]
    header_row = 4
    for idx, h in enumerate(comp_headers, start=1):
        ws_comps.cell(row=header_row, column=idx, value=h)
    _style_header_row(ws_comps, header_row, len(comp_headers))

    start_comp_row = 5
    input_cols = {3, 4, 5, 6, 7, 9, 10, 14}

    for i, (_, row) in enumerate(ordered_comps.iterrows()):
        r = start_comp_row + i
        is_tgt = bool(row["is_target"])
        ws_comps.cell(row=r, column=1, value=row["ticker"]).font = FONT_BOLD if is_tgt else FONT_INPUT
        ws_comps.cell(row=r, column=2, value=row["company_name"]).font = FONT_INPUT
        ws_comps.cell(row=r, column=3, value=_excel_num(row["share_price"])).number_format = FMT_PRICE
        ws_comps.cell(row=r, column=4, value=_excel_num(row["market_cap"], 1e6)).number_format = FMT_CURRENCY_M
        ws_comps.cell(row=r, column=5, value=_excel_num(row["total_debt"], 1e6, 0.0)).number_format = FMT_CURRENCY_M
        ws_comps.cell(row=r, column=6, value=_excel_num(row["minority_interest"], 1e6, 0.0)).number_format = FMT_CURRENCY_M
        ws_comps.cell(row=r, column=7, value=_excel_num(row["cash"], 1e6, 0.0)).number_format = FMT_CURRENCY_M

        ev_cell = ws_comps.cell(row=r, column=8, value=f"=D{r}+E{r}+F{r}-G{r}")
        ev_cell.font = FONT_FORMULA
        ev_cell.number_format = FMT_CURRENCY_M

        ws_comps.cell(row=r, column=9, value=_excel_num(row["ltm_revenue"], 1e6)).number_format = FMT_CURRENCY_M
        ws_comps.cell(row=r, column=10, value=_excel_num(row["ltm_ebitda"], 1e6)).number_format = FMT_CURRENCY_M

        margin_cell = ws_comps.cell(row=r, column=11, value=f'=IF(I{r}>0,J{r}/I{r},"")')
        margin_cell.font = FONT_FORMULA
        margin_cell.number_format = FMT_PERCENT

        ev_rev_cell = ws_comps.cell(row=r, column=12, value=f'=IF(I{r}>0,H{r}/I{r},"")')
        ev_rev_cell.font = FONT_FORMULA
        ev_rev_cell.number_format = FMT_MULTIPLE

        ev_ebitda_cell = ws_comps.cell(row=r, column=13, value=f'=IF(J{r}>0,H{r}/J{r},"")')
        ev_ebitda_cell.font = FONT_FORMULA
        ev_ebitda_cell.number_format = FMT_MULTIPLE

        fpe_cell = ws_comps.cell(row=r, column=14, value=_excel_num(row["forward_pe"]))
        fpe_cell.font = FONT_INPUT
        fpe_cell.number_format = FMT_MULTIPLE

        for c in range(1, len(comp_headers) + 1):
            cell = ws_comps.cell(row=r, column=c)
            cell.border = GRID_BORDER
            if is_tgt:
                cell.fill = SUBHEADER_FILL
            if c in input_cols:
                cell.font = FONT_INPUT

    end_comp_row = start_comp_row + len(ordered_comps) - 1
    peer_start = start_comp_row + 1

    stat_defs = [
        ("Peer 25th Percentile", "PERCENTILE", 0.25),
        ("Peer Median", "MEDIAN", None),
        ("Peer Average", "AVERAGE", None),
        ("Peer 75th Percentile", "PERCENTILE", 0.75),
    ]
    stat_row_map = {}
    summary_cols = [11, 12, 13, 14]

    for idx, (label, func, arg) in enumerate(stat_defs):
        r = end_comp_row + 2 + idx
        stat_row_map[label] = r
        ws_comps.cell(row=r, column=2, value=label).font = FONT_BOLD
        for c in summary_cols:
            col_let = get_column_letter(c)
            if peer_start > end_comp_row:
                formula = None
            elif arg is not None:
                formula = f"={func}({col_let}{peer_start}:{col_let}{end_comp_row},{arg})"
            else:
                formula = f"={func}({col_let}{peer_start}:{col_let}{end_comp_row})"
            cell = ws_comps.cell(row=r, column=c, value=formula)
            cell.font = FONT_BOLD
            cell.border = GRID_BORDER
            cell.fill = HIGHLIGHT_FILL
            cell.number_format = FMT_PERCENT if c == 11 else FMT_MULTIPLE

    note_start = end_comp_row + 7
    ws_comps.cell(
        row=note_start,
        column=1,
        value="Notes: (1) Enterprise Value includes noncontrolling (minority) interest because consolidated EBITDA reflects 100% of joint-venture hospital operations. "
              "(2) Negative/not-meaningful P/E multiples (e.g., CYH) are left blank and automatically excluded by Excel AVERAGE/MEDIAN/PERCENTILE functions. "
              "(3) HCA trades at a scale and margin premium to smaller/higher-leverage peers (THC, UHS, CYH).",
    ).font = FONT_NOTE

    # =========================================================================
    # TAB 2 (PART A): WACC SCHEDULE & BULL / BASE / BEAR SCENARIO TOGGLE
    # =========================================================================

    wacc_info = dcf_results["wacc_schedule"]
    assump = dcf_results["assumptions"]
    proj_df = dcf_results["projections"]

    ws_dcf["A1"] = f"{TARGET_TICKER} — Unlevered Discounted Cash Flow (DCF) Valuation"
    ws_dcf["A1"].font = FONT_TITLE
    ws_dcf["A2"] = (
        f"Valuation Date: {run_date_str} | Last Actual Fiscal Year: FY{last_actual_fy} ({last_actual_dt}) | "
        f"Discounting Convention: End-of-Year Periods 1–{len(proj_df)} | Units: $ in Millions, except per share"
    )
    ws_dcf["A2"].font = FONT_SUBTITLE

    ws_dcf["A4"] = "WACC Calculation"
    ws_dcf["A4"].font = FONT_SECTION
    wacc_rows = [
        ("Risk-Free Rate (10Y UST)", wacc_info["risk_free_rate"], FMT_WACC, FONT_INPUT),
        ("Equity Risk Premium", wacc_info["equity_risk_premium"], FMT_WACC, FONT_INPUT),
        ("Levered Beta", wacc_info["beta"], "0.00", FONT_INPUT),
        ("Cost of Equity (CAPM)", "=B5+B7*B6", FMT_WACC, FONT_FORMULA),
        ("Pre-Tax Cost of Debt", wacc_info["cost_of_debt_pretax"], FMT_WACC, FONT_INPUT),
        ("Effective Tax Rate", wacc_info["effective_tax_rate"], FMT_PERCENT, FONT_INPUT),
        ("After-Tax Cost of Debt", "=B9*(1-B10)", FMT_WACC, FONT_FORMULA),
        ("Equity Weight", wacc_info["weight_equity"], FMT_PERCENT, FONT_INPUT),
        ("Debt Weight", "=1-B12", FMT_PERCENT, FONT_FORMULA),
        ("Computed WACC", "=(B8*B12)+(B11*B13)", FMT_WACC, FONT_BOLD),
    ]
    for offset, (lbl, val, num_fmt, fnt) in enumerate(wacc_rows):
        r = 5 + offset
        ws_dcf.cell(row=r, column=1, value=lbl).font = FONT_BOLD if lbl == "Computed WACC" else FONT_FORMULA
        c = ws_dcf.cell(row=r, column=2, value=val)
        c.font = fnt
        c.number_format = num_fmt
        c.border = TOP_THIN_BOTTOM_DOUBLE if lbl == "Computed WACC" else GRID_BORDER

    # Computed WACC is now in B14
    ws_dcf["G4"] = "Scenario Control (Bull / Base / Bear)"
    ws_dcf["G4"].font = FONT_SECTION

    ws_dcf["G5"] = "Active Scenario"
    ws_dcf["G5"].font = FONT_BOLD
    ws_dcf["G5"].border = GRID_BORDER

    toggle_cell = ws_dcf["H5"]
    toggle_cell.value = "Base"
    toggle_cell.font = FONT_TOGGLE
    toggle_cell.fill = TOGGLE_FILL
    toggle_cell.border = GRID_BORDER
    toggle_cell.alignment = Alignment(horizontal="center")

    dv = DataValidation(type="list", formula1='"Bull,Base,Bear"', allow_blank=False)
    dv.promptTitle = "Select DCF Scenario"
    dv.prompt = "Choose Bull, Base, or Bear to update forecast assumptions."
    ws_dcf.add_data_validation(dv)
    dv.add(toggle_cell)

    case_headers = ["Driver", "Bull", "Base", "Bear"]
    for idx, ch in enumerate(case_headers, start=7):
        cell = ws_dcf.cell(row=6, column=idx, value=ch)
        cell.fill = HEADER_FILL
        cell.font = FONT_HEADER
        cell.alignment = Alignment(horizontal="center")

    base_rev = float(assump["revenue_growth"])
    base_ebitda = float(assump["ebitda_margin"])
    base_dna = float(assump["dna_pct_rev"])
    base_capex = float(assump["capex_pct_rev"])
    base_nwc = float(assump["nwc_pct_delta_rev"])
    base_tax = float(assump["tax_rate"])
    base_tgr = float(assump["terminal_growth_rate"])

    scenario_matrix = [
        ("Revenue Growth", base_rev + 0.015, base_rev, max(-0.50, base_rev - 0.015), FMT_PERCENT),
        ("EBITDA Margin", min(0.95, base_ebitda + 0.010), base_ebitda, max(-0.50, base_ebitda - 0.010), FMT_PERCENT),
        ("D&A (% of Rev)", base_dna, base_dna, base_dna, FMT_PERCENT),
        ("CapEx (% of Rev)", max(0.0, base_capex - 0.0025), base_capex, min(0.95, base_capex + 0.0025), FMT_PERCENT),
        ("NWC (% of dRev)", base_nwc, base_nwc, base_nwc, FMT_PERCENT),
        ("Cash Tax Rate", base_tax, base_tax, base_tax, FMT_PERCENT),
        ("Terminal Growth (g)", min(0.06, base_tgr + 0.005), base_tgr, max(0.0, base_tgr - 0.005), FMT_WACC),
        ("Exit EV/EBITDA", f"='Trading Comps'!M{stat_row_map['Peer 75th Percentile']}", f"='Trading Comps'!M{stat_row_map['Peer Median']}", f"='Trading Comps'!M{stat_row_map['Peer 25th Percentile']}", FMT_MULTIPLE),
    ]
    for offset, (drv_lbl, bull_v, base_v, bear_v, num_fmt) in enumerate(scenario_matrix):
        r = 7 + offset
        ws_dcf.cell(row=r, column=7, value=drv_lbl).font = FONT_FORMULA
        ws_dcf.cell(row=r, column=7).border = GRID_BORDER
        for col_offset, val in enumerate((bull_v, base_v, bear_v)):
            c = ws_dcf.cell(row=r, column=8 + col_offset, value=val)
            c.font = FONT_LINK if isinstance(val, str) and val.startswith("=") else FONT_INPUT
            c.number_format = num_fmt
            c.border = GRID_BORDER

    ws_dcf["D4"] = "Active Forecast Drivers"
    ws_dcf["D4"].font = FONT_SECTION
    active_driver_labels = [
        ("Projected Revenue Growth", FMT_PERCENT),
        ("Projected EBITDA Margin", FMT_PERCENT),
        ("D&A (% of Revenue)", FMT_PERCENT),
        ("CapEx (% of Revenue)", FMT_PERCENT),
        ("NWC (% of Delta Revenue)", FMT_PERCENT),
        ("Cash Tax Rate", FMT_PERCENT),
        ("Terminal Perpetuity Growth (g)", FMT_WACC),
        ("Exit EV/EBITDA Multiple", FMT_MULTIPLE),
    ]
    for offset, (lbl, num_fmt) in enumerate(active_driver_labels):
        r = 5 + offset
        src_r = 7 + offset
        ws_dcf.cell(row=r, column=4, value=lbl).font = FONT_FORMULA
        c = ws_dcf.cell(row=r, column=5, value=f"=INDEX(H{src_r}:J{src_r},1,MATCH($H$5,$H$6:$J$6,0))")
        c.font = FONT_BOLD
        c.fill = HIGHLIGHT_FILL
        c.number_format = num_fmt
        c.border = GRID_BORDER

    # =========================================================================
    # TAB 2 (PART B): HISTORICAL & PROJECTED UFCF + VALUATION BRIDGE
    # =========================================================================
    ws_dcf["A16"] = "Historical & Projected Unlevered Free Cash Flow ($M)"
    ws_dcf["A16"].font = FONT_SECTION

    all_years = list(hist_df["fiscal_year"]) + list(proj_df["fiscal_year"])
    ws_dcf.cell(row=17, column=1, value="Line Item ($M)")
    for idx, yr in enumerate(all_years, start=2):
        ws_dcf.cell(row=17, column=idx, value=yr)
    _style_header_row(ws_dcf, 17, len(all_years) + 1)

    dcf_line_labels = [
        "Total Revenue", "EBITDA", "Less: D&A", "EBIT (Operating Income)",
        "Less: Cash Taxes", "NOPAT", "Plus: D&A", "Less: CapEx",
        "Less: Change in NWC", "Unlevered Free Cash Flow (UFCF)",
        "Discount Period (Years)", "Discount Factor", "Present Value of UFCF",
    ]
    for offset, lbl in enumerate(dcf_line_labels):
        cell = ws_dcf.cell(row=18 + offset, column=1, value=lbl)
        cell.font = FONT_BOLD if "UFCF" in lbl or lbl == "NOPAT" else FONT_FORMULA

    num_hist = len(hist_df)
    for i, (_, hrow) in enumerate(hist_df.iterrows()):
        col = 2 + i
        col_let = get_column_letter(col)
        ws_dcf.cell(row=18, column=col, value=_excel_num(hrow["revenue"], 1e6, 0.0)).font = FONT_INPUT
        ws_dcf.cell(row=19, column=col, value=_excel_num(hrow["ebitda"], 1e6, 0.0)).font = FONT_INPUT
        ws_dcf.cell(row=20, column=col, value=_excel_num(hrow["dna"], 1e6, 0.0)).font = FONT_INPUT
        ws_dcf.cell(row=21, column=col, value=f"={col_let}19-{col_let}20").font = FONT_FORMULA
        ws_dcf.cell(row=22, column=col, value=f"={col_let}21*{float(hrow['tax_rate']):.4f}").font = FONT_FORMULA
        ws_dcf.cell(row=23, column=col, value=f"={col_let}21-{col_let}22").font = FONT_BOLD
        ws_dcf.cell(row=24, column=col, value=f"={col_let}20").font = FONT_FORMULA
        ws_dcf.cell(row=25, column=col, value=_excel_num(hrow["capex"], 1e6, 0.0)).font = FONT_INPUT
        ws_dcf.cell(row=26, column=col, value=_excel_num(hrow["delta_nwc"], 1e6, 0.0)).font = FONT_INPUT
        ws_dcf.cell(row=27, column=col, value=f"={col_let}23+{col_let}24-{col_let}25-{col_let}26").font = FONT_BOLD

        for r_fmt in range(18, 28):
            ws_dcf.cell(row=r_fmt, column=col).number_format = FMT_CURRENCY_M
            ws_dcf.cell(row=r_fmt, column=col).border = GRID_BORDER

    currency_proj_rows = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30]
    for j in range(len(proj_df)):
        col = 2 + num_hist + j
        col_let = get_column_letter(col)
        prev_let = get_column_letter(col - 1)
        period_num = j + 1

        ws_dcf.cell(row=18, column=col, value=f"={prev_let}18*(1+$E$5)").font = FONT_FORMULA
        ws_dcf.cell(row=19, column=col, value=f"={col_let}18*$E$6").font = FONT_FORMULA
        ws_dcf.cell(row=20, column=col, value=f"={col_let}18*$E$7").font = FONT_FORMULA
        ws_dcf.cell(row=21, column=col, value=f"={col_let}19-{col_let}20").font = FONT_FORMULA
        ws_dcf.cell(row=22, column=col, value=f"={col_let}21*$E$10").font = FONT_FORMULA
        ws_dcf.cell(row=23, column=col, value=f"={col_let}21-{col_let}22").font = FONT_BOLD
        ws_dcf.cell(row=24, column=col, value=f"={col_let}20").font = FONT_FORMULA
        ws_dcf.cell(row=25, column=col, value=f"={col_let}18*$E$8").font = FONT_FORMULA
        ws_dcf.cell(row=26, column=col, value=f"=({col_let}18-{prev_let}18)*$E$9").font = FONT_FORMULA
        ws_dcf.cell(row=27, column=col, value=f"={col_let}23+{col_let}24-{col_let}25-{col_let}26").font = FONT_BOLD
        ws_dcf.cell(row=28, column=col, value=period_num).font = FONT_INPUT
        ws_dcf.cell(row=29, column=col, value=f"=1/((1+$B$14)^{col_let}28)").font = FONT_FORMULA
        ws_dcf.cell(row=30, column=col, value=f"={col_let}27*{col_let}29").font = FONT_BOLD

        for r_fmt in currency_proj_rows:
            ws_dcf.cell(row=r_fmt, column=col).number_format = FMT_CURRENCY_M
            ws_dcf.cell(row=r_fmt, column=col).border = GRID_BORDER
        ws_dcf.cell(row=28, column=col).number_format = FMT_INT
        ws_dcf.cell(row=28, column=col).border = GRID_BORDER
        ws_dcf.cell(row=29, column=col).number_format = "0.0000"
        ws_dcf.cell(row=29, column=col).border = GRID_BORDER

    first_proj_col = get_column_letter(2 + num_hist)
    last_proj_col = get_column_letter(1 + num_hist + len(proj_df))
    proj_span = f"{proj_df['fiscal_year'].iloc[0]}–{proj_df['fiscal_year'].iloc[-1]}"

    ws_dcf["A32"] = "Valuation Bridge ($M, except per share)"
    ws_dcf["A32"].font = FONT_SECTION
    ws_dcf["B32"] = "Perpetuity Growth"
    ws_dcf["C32"] = "Exit Multiple"
    _style_header_row(ws_dcf, 32, 3)

    bridge_rows = [
        (f"Sum of PV of UFCF ({proj_span})", f"=SUM({first_proj_col}30:{last_proj_col}30)", f"=SUM({first_proj_col}30:{last_proj_col}30)", FMT_CURRENCY_M, FONT_FORMULA),
        ("Terminal Value (Undiscounted)", f"=({last_proj_col}27*(1+$E$11))/($B$14-$E$11)", f'=IF({last_proj_col}19>0,{last_proj_col}19*$E$12,"")', FMT_CURRENCY_M, FONT_FORMULA),
        ("Present Value of Terminal Value", f"=B34*{last_proj_col}29", f'=IF(ISNUMBER(C34),C34*{last_proj_col}29,"")', FMT_CURRENCY_M, FONT_FORMULA),
        ("Implied Enterprise Value (EV)", "=B33+B35", '=IF(ISNUMBER(C35),C33+C35,"")', FMT_CURRENCY_M, FONT_BOLD),
        ("Terminal Value (% of EV)", "=B35/B36", '=IF(ISNUMBER(C36),C35/C36,"")', FMT_PERCENT, FONT_FORMULA),
        ("(+) Cash & Equivalents", "='Trading Comps'!G5", "='Trading Comps'!G5", FMT_CURRENCY_M, FONT_LINK),
        ("(-) Total Debt", "='Trading Comps'!E5", "='Trading Comps'!E5", FMT_CURRENCY_M, FONT_LINK),
        ("(-) Minority Interest", "='Trading Comps'!F5", "='Trading Comps'!F5", FMT_CURRENCY_M, FONT_LINK),
        ("Implied Equity Value", "=B36+B38-B39-B40", '=IF(ISNUMBER(C36),C36+C38-C39-C40,"")', FMT_CURRENCY_M, FONT_BOLD),
        ("Diluted Shares Outstanding (M)", _excel_num(target_row["shares_outstanding"], 1e6), _excel_num(target_row["shares_outstanding"], 1e6), "0.0", FONT_INPUT),
        ("Implied Share Price ($)", "=B41/B42", '=IF(ISNUMBER(C41),C41/C42,"")', FMT_PRICE, FONT_BOLD),
    ]

    for offset, (lbl, f_perp, f_exit, num_fmt, fnt) in enumerate(bridge_rows):
        r = 33 + offset
        ws_dcf.cell(row=r, column=1, value=lbl).font = FONT_BOLD if "Implied" in lbl else FONT_FORMULA
        c1 = ws_dcf.cell(row=r, column=2, value=f_perp)
        c2 = ws_dcf.cell(row=r, column=3, value=f_exit)
        for c in (c1, c2):
            c.font = fnt
            c.number_format = num_fmt
            c.border = TOP_THIN_BOTTOM_DOUBLE if "Implied Share Price" in lbl else GRID_BORDER
            if "Implied Share Price" in lbl:
                c.fill = SUBHEADER_FILL

    # =========================================================================
    # TAB 3: SENSITIVITY ANALYSIS (Exact Base WACC Linkage -> Zero Rounding Drift)
    # =========================================================================
    ws_sens["A1"] = f"{TARGET_TICKER} — Implied Share Price Sensitivity Analysis ($ / Share)"
    ws_sens["A1"].font = FONT_TITLE
    ws_sens["A2"] = (
        f"Center row links directly to Computed WACC ('DCF Model'!B14 = {wacc_info['wacc']:.2%}) "
        f"so center cell matches DCF Model output to the exact cent."
    )
    ws_sens["A2"].font = FONT_SUBTITLE

    wacc_offsets = [-0.010, -0.005, 0.0, 0.005, 0.010]
    tgr_grid = [0.015, 0.020, 0.025, 0.030, 0.035]
    n_periods = len(proj_df)

    ufcf_range = f"'DCF Model'!${first_proj_col}$27:${last_proj_col}$27"
    term_ufcf_ref = f"'DCF Model'!${last_proj_col}$27"
    term_ebitda_ref = f"'DCF Model'!${last_proj_col}$19"

    ws_sens["A4"] = "Table 1: WACC (Rows) vs. Terminal Growth Rate (Columns)"
    ws_sens["A4"].font = FONT_SECTION
    ws_sens["A5"] = "WACC \\ TGR"
    ws_sens["A5"].font = FONT_HEADER
    ws_sens["A5"].fill = HEADER_FILL

    for c_idx, g_val in enumerate(tgr_grid, start=2):
        cell = ws_sens.cell(row=5, column=c_idx, value=g_val)
        cell.font = FONT_HEADER
        cell.fill = HEADER_FILL
        cell.number_format = FMT_WACC
        cell.alignment = Alignment(horizontal="center")

    for r_idx, step in enumerate(wacc_offsets, start=6):
        w_formula = "='DCF Model'!$B$14" if step == 0.0 else f"='DCF Model'!$B$14+({step})"
        w_cell = ws_sens.cell(row=r_idx, column=1, value=w_formula)
        w_cell.font = FONT_BOLD if step == 0.0 else FONT_LINK
        w_cell.fill = SUBHEADER_FILL
        w_cell.number_format = FMT_WACC

        for c_idx, g_val in enumerate(tgr_grid, start=2):
            col_let = get_column_letter(c_idx)
            formula = (
                f"=IF($A{r_idx}>{col_let}$5,"
                f"(NPV($A{r_idx},{ufcf_range})+(({term_ufcf_ref}*(1+{col_let}$5))/($A{r_idx}-{col_let}$5))/((1+$A{r_idx})^{n_periods})"
                f"+'DCF Model'!$B$38-'DCF Model'!$B$39-'DCF Model'!$B$40)/'DCF Model'!$B$42,\"\")"
            )

            cell = ws_sens.cell(row=r_idx, column=c_idx, value=formula)
            cell.number_format = FMT_PRICE
            cell.border = GRID_BORDER
            is_base = (step == 0.0) and abs(g_val - assump["terminal_growth_rate"]) < 1e-5
            cell.font = FONT_BOLD if is_base else FONT_FORMULA
            if is_base:
                cell.fill = SUBHEADER_FILL

    ws_sens["A13"] = "Table 2: WACC (Rows) vs. Exit EV/EBITDA Multiple (Columns)"
    ws_sens["A13"].font = FONT_SECTION
    ws_sens["A14"] = "WACC \\ Exit Mult"
    ws_sens["A14"].font = FONT_HEADER
    ws_sens["A14"].fill = HEADER_FILL

    mult_grid = [6.0, 7.0, 8.0, 9.0, 10.0]
    for c_idx, m_val in enumerate(mult_grid, start=2):
        cell = ws_sens.cell(row=14, column=c_idx, value=m_val)
        cell.font = FONT_HEADER
        cell.fill = HEADER_FILL
        cell.number_format = FMT_MULTIPLE
        cell.alignment = Alignment(horizontal="center")

    for r_idx, step in enumerate(wacc_offsets, start=15):
        w_formula = "='DCF Model'!$B$14" if step == 0.0 else f"='DCF Model'!$B$14+({step})"
        w_cell = ws_sens.cell(row=r_idx, column=1, value=w_formula)
        w_cell.font = FONT_BOLD if step == 0.0 else FONT_LINK
        w_cell.fill = SUBHEADER_FILL
        w_cell.number_format = FMT_WACC

        for c_idx, m_val in enumerate(mult_grid, start=2):
            col_let = get_column_letter(c_idx)
            formula = (
                f"=IF({term_ebitda_ref}>0,"
                f"(NPV($A{r_idx},{ufcf_range})+({term_ebitda_ref}*{col_let}$14)/((1+$A{r_idx})^{n_periods})"
                f"+'DCF Model'!$B$38-'DCF Model'!$B$39-'DCF Model'!$B$40)/'DCF Model'!$B$42,\"\")"
            )

            cell = ws_sens.cell(row=r_idx, column=c_idx, value=formula)
            cell.number_format = FMT_PRICE
            cell.border = GRID_BORDER
            is_base_mult = (step == 0.0) and abs(m_val - assump["exit_ebitda_multiple"]) < 0.05
            cell.font = FONT_BOLD if is_base_mult else FONT_FORMULA
            if is_base_mult:
                cell.fill = SUBHEADER_FILL

    # =========================================================================
    # TAB 1: EXECUTIVE SUMMARY & TRUE FLOATING-BAR FOOTBALL FIELD CHART
    # =========================================================================
    ws_exec["A1"] = f"{TARGET_TICKER} ({target_row['company_name']}) — Valuation Executive Summary"
    ws_exec["A1"].font = FONT_TITLE
    ws_exec["A2"] = (
        f"Valuation Date: {run_date_str} | Historical Anchor: FY{last_actual_fy} ({last_actual_dt}) | "
        f"DCF Ranges Reflect WACC +/- 50 bps; Comps Ranges Reflect Peer 25th–75th Percentiles"
    )
    ws_exec["A2"].font = FONT_SUBTITLE

    ws_exec["A4"] = "Valuation Football Field Summary ($ / Share)"
    ws_exec["A4"].font = FONT_SECTION

    ff_headers = [
        "Valuation Methodology",
        "Low ($)",
        "Spread (High - Low) ($)",
        "Base / Mid ($)",
        "High ($)",
        "Current Price ($)",
    ]
    for idx, h in enumerate(ff_headers, start=1):
        ws_exec.cell(row=5, column=idx, value=h)
    _style_header_row(ws_exec, 5, len(ff_headers))

    for i, (_, frow) in enumerate(football_field_df.iterrows()):
        r = 6 + i
        ws_exec.cell(row=r, column=1, value=frow["methodology"]).font = FONT_BOLD
        ws_exec.cell(row=r, column=2, value=_excel_num(frow["low_price"])).number_format = FMT_PRICE

        spread_cell = ws_exec.cell(row=r, column=3, value=f'=IF(AND(ISNUMBER(E{r}),ISNUMBER(B{r})),E{r}-B{r},"")')
        spread_cell.font = FONT_FORMULA
        spread_cell.number_format = FMT_PRICE

        ws_exec.cell(row=r, column=4, value=_excel_num(frow["mid_price"])).number_format = FMT_PRICE
        ws_exec.cell(row=r, column=5, value=_excel_num(frow["high_price"])).number_format = FMT_PRICE

        cur_cell = ws_exec.cell(row=r, column=6, value="='Trading Comps'!C5")
        cur_cell.font = FONT_LINK
        cur_cell.number_format = FMT_PRICE

        for c in range(1, 7):
            ws_exec.cell(row=r, column=c).border = GRID_BORDER

    # Add a 5th comparison row for Current Market Price so it plots directly on the Football Field
    mkt_row = 6 + len(football_field_df)
    ws_exec.cell(row=mkt_row, column=1, value="Current Market Price (Trading)").font = FONT_BOLD
    ws_exec.cell(row=mkt_row, column=2, value="='Trading Comps'!C5").number_format = FMT_PRICE
    ws_exec.cell(row=mkt_row, column=3, value=4.0).number_format = FMT_PRICE  # Thin marker bar
    ws_exec.cell(row=mkt_row, column=4, value="='Trading Comps'!C5").number_format = FMT_PRICE
    ws_exec.cell(row=mkt_row, column=5, value="='Trading Comps'!C5").number_format = FMT_PRICE
    ws_exec.cell(row=mkt_row, column=6, value="='Trading Comps'!C5").number_format = FMT_PRICE
    for c in range(1, 7):
        cell = ws_exec.cell(row=mkt_row, column=c)
        cell.border = GRID_BORDER
        cell.fill = TOGGLE_FILL
        if c >= 2:
            cell.font = FONT_LINK

    chart = BarChart()
    chart.type = "bar"
    chart.grouping = "stacked"
    chart.overlap = 100
    chart.style = 10
    chart.title = f"{TARGET_TICKER} Implied Share Price Football Field vs. Current Market Price ($ / Share)"
    chart.x_axis.title = "Valuation Methodology"
    chart.y_axis.title = "Share Price ($)"
    chart.width = 19
    chart.height = 11

    data_ref = Reference(ws_exec, min_col=2, max_col=3, min_row=5, max_row=mkt_row)
    cats_ref = Reference(ws_exec, min_col=1, min_row=6, max_row=mkt_row)
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)

    if len(chart.series) >= 2:
        base_series, range_series = chart.series[:2]
        base_series.graphicalProperties.solidFill = "FFFFFF"
        base_series.graphicalProperties.line.noFill = True

        range_series.graphicalProperties.solidFill = "1F4E78"
        range_series.graphicalProperties.line.solidFill = "1F4E78"

    chart.legend = None
    ws_exec.add_chart(chart, "A13")

    ws_exec.cell(
        row=33,
        column=1,
        value="Methodology Notes: (1) DCF Perpetuity Growth ($497/sh mid) aligns closely with HCA's market price ($476/sh). "
              "(2) Exit Multiple DCF and Trading Comps sit lower because HCA's peers (THC, UHS, CYH) trade at distressed/scale-discounted multiples (5.4x–8.1x vs. HCA's 9.9x). "
              "(3) EV / Revenue is shown for reference only and should not be equal-weighted because it ignores HCA's superior EBITDA margin profile.",
    ).font = FONT_NOTE

    for ws in [ws_exec, ws_dcf, ws_sens, ws_comps]:
        _autofit_columns(ws)

    output_path = OUTPUT_DIR / output_filename
    wb.save(output_path)
    logger.info("Successfully generated Excel valuation model: %s", output_path)
    return output_path

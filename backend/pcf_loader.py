"""PCF (portfolio composition / basket) download and parsing.

The PCF is the published list of exactly what one creation unit of the ETF holds.
We download it live from the issuer, parse the issuer-specific layout into one
standardised shape, keep only the Treasury rows, and capture the cash component
separately because cash flows straight into NAV without being priced.

Standardised output columns, identical regardless of issuer:
    cusip, isin, name, par_value, weight, maturity, coupon

All downloads use the shared three-attempt, five second retry policy.
"""

from __future__ import annotations

import io
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console

from . import etf_config

console = Console()

# Local holdings fallback. If a CSV for a ticker is present here it is used in
# preference to a live fetch, so a manually downloaded basket keeps the tool
# working when a provider CDN blocks this host. Files are keyed by ticker, e.g.
# data/GOVT.csv or data/GOVT_holdings.csv.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class PCFDownloadError(Exception):
    """Raised when a PCF download is blocked or returns HTML instead of CSV.

    Carries the provider and ticker so the failure is unambiguous regardless of
    which issuer's CDN did the blocking.
    """


# Realistic browser headers, centralised so every PCF download path for every
# provider sends the same convincing request. Provider CDNs (Akamai and similar)
# reject requests that do not look like a real browser, returning 403 Forbidden
# or an HTML bot-challenge page. A current Chrome-on-macOS User-Agent plus the
# usual Accept, Accept-Language, and a same-site Referer make the request look
# like a normal browser navigation from the provider's own site.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Per-provider Referer pointing at that provider's own product or fund page, so
# the download looks like it was initiated from the issuer's site. All three
# providers use the identical header approach; only the Referer host differs.
PROVIDER_REFERERS: dict[str, str] = {
    "ishares": "https://www.ishares.com/us/products/etf-investments",
    "vanguard": "https://investor.vanguard.com/investment-products/etfs/profile",
    "ssga": "https://www.ssga.com/us/en/intermediary/etfs/fund-finder",
}


def browser_headers(provider: str, referer: str | None = None) -> dict[str, str]:
    """Return realistic browser headers for a provider PCF download.

    Centralised so iShares, Vanguard, and State Street (SSGA) all send the same
    convincing browser fingerprint. Includes a current Chrome-on-macOS
    User-Agent, an Accept and Accept-Language a browser would send, and a Referer
    set to the provider's own product or fund page (overridable per request, for
    example with a specific iShares product page). The Sec-Fetch hints mirror what
    Chrome sends on a same-site download navigation.
    """
    return {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "text/csv,application/csv,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": referer or PROVIDER_REFERERS.get(provider, ""),
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
    }


# iShares serves holdings CSVs from product pages following this pattern. The
# numeric id and slug identify the specific fund; they come from the routing
# table once a fund has been discovered.
ISHARES_URL_PATTERN = (
    "https://www.ishares.com/us/products/{product_id}/{slug}/"
    "1467271812596.ajax?fileType=csv&fileName={ticker}_holdings&dataType=fund"
)

# Known iShares product ids and slugs for the starting watchlist. Discovery can
# extend this, but the watchlist funds are pinned here so the PCF URL is
# deterministic.
ISHARES_PRODUCTS: dict[str, dict[str, str]] = {
    "IEF": {"product_id": "239456", "slug": "ishares-7-10-year-treasury-bond-etf"},
    "SHY": {"product_id": "239452", "slug": "ishares-1-3-year-treasury-bond-etf"},
    "IEI": {"product_id": "239455", "slug": "ishares-3-7-year-treasury-bond-etf"},
    "TLT": {"product_id": "239454", "slug": "ishares-20-year-treasury-bond-etf"},
    "GOVT": {"product_id": "239468", "slug": "ishares-us-treasury-bond-etf"},
}

# Substrings that mark a holding row as something other than a priceable
# Treasury: cash, futures, repo, and money market sweeps. These are dropped from
# the basket and the cash-like ones are summed into the cash component.
NON_TREASURY_MARKERS = (
    "cash",
    "usd",
    "future",
    "repo",
    "money market",
    "net other",
    "margin",
)


@dataclass
class ParsedPCF:
    """A parsed basket: priceable Treasury holdings plus the cash component."""

    holdings: pd.DataFrame  # standardised columns
    cash_component: float  # US dollars of cash in one creation unit
    as_of: date | None = None
    raw_row_count: int = 0
    dropped: list[str] = field(default_factory=list)


def _provider_referer(provider: str, ticker: str) -> str:
    """Return the best Referer for a provider download.

    For iShares the specific product page is used when the ticker is known, which
    is the most convincing same-site Referer. Other providers fall back to their
    configured fund-finder page.
    """
    if provider == "ishares":
        product = ISHARES_PRODUCTS.get(ticker.upper())
        if product:
            return (
                f"https://www.ishares.com/us/products/{product['product_id']}/"
                f"{product['slug']}"
            )
    return PROVIDER_REFERERS.get(provider, "")


def _looks_like_html(text: str) -> bool:
    """True if a response body is an HTML page rather than CSV.

    Provider CDNs answer a blocked request with an HTML bot-challenge or consent
    page. We sniff the leading bytes for an HTML marker so we never hand megabytes
    of HTML to the CSV parser.
    """
    head = text.lstrip()[:256].lower()
    return (
        head.startswith("<!doctype html")
        or head.startswith("<html")
        or head.startswith("<?xml")
        and "<html" in head
        or "<title" in head
    )


def _retry_download(url: str, provider: str, ticker: str) -> str:
    """Download a PCF as text with realistic browser headers and retries.

    The same centralised browser headers are used for every provider (iShares,
    Vanguard, SSGA); only the Referer host differs. Transient network failures
    (timeouts, connection errors, 5xx) are retried under the shared policy. A 403
    Forbidden or an HTML bot-challenge page is a definitive block that retrying
    will not fix, so it raises a clear PCFDownloadError naming the provider and
    ticker immediately rather than burning retries or parsing HTML.
    """
    headers = browser_headers(provider, _provider_referer(provider, ticker))
    what = f"PCF download for {ticker} ({provider})"
    last_exc: Exception | None = None
    for attempt in range(1, etf_config.DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=30, headers=headers)
            if resp.status_code == 403:
                raise PCFDownloadError(
                    f"{provider} returned 403 Forbidden for {ticker}. The provider "
                    f"CDN is blocking this host as a bot. Retry from a server whose "
                    f"IP is not bot-filtered, or place a manually downloaded CSV at "
                    f"{DATA_DIR}/{ticker.upper()}.csv."
                )
            resp.raise_for_status()
            text = resp.text
            if _looks_like_html(text):
                raise PCFDownloadError(
                    f"{provider} returned an HTML bot-challenge page, not CSV, for "
                    f"{ticker}. The provider CDN is blocking this host. Retry from a "
                    f"server whose IP is not bot-filtered, or place a manually "
                    f"downloaded CSV at {DATA_DIR}/{ticker.upper()}.csv."
                )
            return text
        except PCFDownloadError:
            # A definitive block; retrying will not help, so surface it now.
            raise
        except Exception as exc:  # noqa: BLE001 - transient network errors retry
            last_exc = exc
            console.log(
                f"[yellow]{what} attempt {attempt}/"
                f"{etf_config.DOWNLOAD_RETRIES} failed: {exc}[/yellow]"
            )
            if attempt < etf_config.DOWNLOAD_RETRIES:
                time.sleep(etf_config.DOWNLOAD_RETRY_WAIT_SECONDS)
    assert last_exc is not None
    raise last_exc


def build_pcf_url(ticker: str, provider: str) -> str:
    """Construct the PCF download URL for a ticker from its provider pattern.

    Only iShares is wired with a deterministic pattern for the watchlist. Vanguard
    and SSGA raise until their product identifiers are supplied, since their
    holdings endpoints are not a simple templated URL.
    """
    ticker = ticker.upper()
    if provider == "ishares":
        product = ISHARES_PRODUCTS.get(ticker)
        if not product:
            raise ValueError(
                f"No iShares product mapping for {ticker}; supply product id/slug."
            )
        return ISHARES_URL_PATTERN.format(
            product_id=product["product_id"], slug=product["slug"], ticker=ticker
        )
    raise ValueError(
        f"Provider '{provider}' PCF URL construction not configured for {ticker}."
    )


def _is_treasury(name: str) -> bool:
    """Heuristic: does this holding name look like a US Treasury?

    iShares Treasury funds label rows "TREASURY NOTE" / "TREASURY BOND" /
    "TREASURY BILL". We keep those and drop everything matching a non-Treasury
    marker.
    """
    lower = name.lower()
    if any(marker in lower for marker in NON_TREASURY_MARKERS):
        return False
    return "treasury" in lower or "t-note" in lower or "t-bond" in lower


def _to_float(value) -> float:
    """Parse a number that may carry commas, percent signs, or be blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).replace(",", "").replace("%", "").replace("$", "").strip()
    if text in ("", "-", "--"):
        return 0.0
    return float(text)


def synthetic_id(coupon: float, maturity: date | None) -> str:
    """Build the bond key from coupon and maturity.

    The iShares holdings file carries no CUSIP, so we identify each Treasury by
    the two attributes that uniquely define it on a desk: its coupon and its
    maturity date. Reopenings of the same line share both and are genuinely
    fungible, so collapsing them onto one key is correct. The key looks like
    "UST_4.250_2034-11-15" and is used everywhere CUSIP was used as the bond key
    (bonds table, basket_holdings, the marks join). The coupon is formatted as a
    percentage to three decimals for a stable, readable, collision-free key.
    """
    coupon_pct = f"{coupon * 100:.3f}"
    mat = maturity.isoformat() if maturity else "NA"
    return f"UST_{coupon_pct}_{mat}"


# ---------------------------------------------------------------------------
# SpreadsheetML (real iShares ".xls") parsing
# ---------------------------------------------------------------------------

# The iShares holdings file served as .xls is not a real Excel binary. It is a
# Microsoft SpreadsheetML XML document (it opens with <?xml ...> and
# <ss:Workbook>), so pandas.read_excel cannot read it. We parse it as XML: the
# basket lives in the "Holdings" worksheet as <ss:Row> elements whose values are
# the text of the <ss:Data> cells.
SS_NS = "{urn:schemas-microsoft-com:office:spreadsheet}"

# The 24-column holdings header. We locate the header by matching its leading
# columns, then read rows after it.
ISHARES_HEADER_LEAD = (
    "Name",
    "Sector",
    "Asset Class",
    "Market Value",
    "Weight (%)",
    "Notional Value",
    "Par Value",
)


def _looks_like_spreadsheetml(text: str) -> bool:
    """True if a response body is SpreadsheetML XML rather than CSV or HTML."""
    head = text.lstrip()[:512].lower()
    return "<ss:workbook" in head or "urn:schemas-microsoft-com:office:spreadsheet" in head


def _row_values(row: ET.Element) -> list[str]:
    """Return the ordered cell text of one <ss:Row>, honouring ss:Index gaps.

    SpreadsheetML lets a cell declare ss:Index to skip blank columns, so we pad
    with empty strings to keep every value aligned to its column position.
    """
    values: list[str] = []
    for cell in row.findall(f"{SS_NS}Cell"):
        index = cell.get(f"{SS_NS}Index")
        if index is not None:
            target = int(index) - 1
            while len(values) < target:
                values.append("")
        data = cell.find(f"{SS_NS}Data")
        values.append(data.text if (data is not None and data.text is not None) else "")
    return values


def _extract_holdings_rows(raw_text: str) -> list[list[str]]:
    """Extract the Holdings worksheet rows from the SpreadsheetML document.

    Only the Holdings worksheet is parsed, which both avoids the document's
    malformed Disclaimers sheet (it embeds raw HTML) and excludes the thousands
    of trailing rows that live in the Historical and Distributions worksheets. We
    isolate the worksheet block, wrap it in a minimal workbook, and parse it; if
    strict XML parsing still fails we fall back to a tolerant regex extraction.
    """
    match = re.search(
        r'<ss:Worksheet ss:Name="Holdings">.*?</ss:Worksheet>', raw_text, re.S
    )
    if not match:
        raise PCFDownloadError("iShares SpreadsheetML has no Holdings worksheet")
    block = match.group(0)
    wrapped = (
        '<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        + block
        + "</ss:Workbook>"
    )
    try:
        root = ET.fromstring(wrapped)
        return [_row_values(r) for r in root.iter(f"{SS_NS}Row")]
    except ET.ParseError:
        return _regex_extract_rows(block)


def _regex_extract_rows(block: str) -> list[list[str]]:
    """Tolerant fallback: extract rows and cell text by regex.

    Used only if strict XML parsing of the Holdings worksheet fails. Unescapes
    the standard XML entities and honours ss:Index gaps.
    """
    rows: list[list[str]] = []
    for row_match in re.finditer(r"<ss:Row\b.*?</ss:Row>", block, re.S):
        row_xml = row_match.group(0)
        cells: list[str] = []
        for cell_match in re.finditer(r"<ss:Cell\b([^>]*)>(.*?)</ss:Cell>", row_xml, re.S):
            attrs, inner = cell_match.group(1), cell_match.group(2)
            index = re.search(r'ss:Index="(\d+)"', attrs)
            if index:
                target = int(index.group(1)) - 1
                while len(cells) < target:
                    cells.append("")
            data = re.search(r"<ss:Data\b[^>]*>(.*?)</ss:Data>", inner, re.S)
            text = data.group(1) if data else ""
            for entity, char in (
                ("&amp;", "&"),
                ("&lt;", "<"),
                ("&gt;", ">"),
                ("&quot;", '"'),
                ("&#39;", "'"),
                ("&apos;", "'"),
            ):
                text = text.replace(entity, char)
            cells.append(text.strip())
        rows.append(cells)
    return rows


def parse_ishares_spreadsheetml(raw_text: str) -> ParsedPCF:
    """Parse the real iShares SpreadsheetML holdings file into the basket shape.

    Layout handled:

    - A metadata block of the first rows (a date, the fund name, the inception
      date, a "Fund Holdings as of <DATE>" row, the number of securities, and
      shares outstanding). The "Fund Holdings as of" date becomes the basket
      as-of date.
    - A 24-column header row (Name, Sector, Asset Class, Market Value, Weight (%),
      Notional Value, Par Value, Price, ... Maturity, Coupon (%), ...).
    - Holdings rows after the header. We keep ONLY rows where Asset Class is
      "Fixed Income" and Sector is "Treasuries"; the file's ~207 real holdings are
      followed by trailing rows that do not match and are excluded.

    Column mapping to the standardised schema:
        name      <- Name
        par_value <- Par Value
        weight    <- Weight (%)
        price     <- Price (clean price per 100, kept for validation only; the
                     engine reprices off the curve regardless)
        maturity  <- Maturity ("Nov 15, 2031" style)
        coupon    <- Coupon (%)
    There is no CUSIP in this file, so the bond key is the coupon+maturity
    synthetic identifier. A cash row (Asset Class "Cash") is captured as the cash
    component; otherwise cash defaults to 0.
    """
    rows = _extract_holdings_rows(raw_text)

    # Extract the "Fund Holdings as of" date from the metadata block.
    as_of: date | None = None
    for row in rows:
        if row and row[0] and "fund holdings as of" in row[0].strip().lower():
            if len(row) > 1:
                as_of = _parse_maturity(row[1])
            break

    # Locate the holdings header row by its leading columns.
    header_idx = None
    for i, row in enumerate(rows):
        lead = [c.strip().lower() for c in row[: len(ISHARES_HEADER_LEAD)]]
        if lead == [h.lower() for h in ISHARES_HEADER_LEAD]:
            header_idx = i
            break
    if header_idx is None:
        raise PCFDownloadError("iShares Holdings worksheet has no recognisable header")

    header = [c.strip() for c in rows[header_idx]]
    col_index = {name: i for i, name in enumerate(header)}

    def cell(row: list[str], name: str) -> str:
        i = col_index.get(name)
        return row[i].strip() if (i is not None and i < len(row)) else ""

    holdings_rows = []
    cash_component = 0.0
    dropped: list[str] = []

    for row in rows[header_idx + 1 :]:
        if not any(c.strip() for c in row):
            continue
        asset_class = cell(row, "Asset Class")
        sector = cell(row, "Sector")
        name = cell(row, "Name")
        # Keep only genuine Treasury holdings.
        if asset_class == "Fixed Income" and sector == "Treasuries":
            maturity = _parse_maturity(cell(row, "Maturity"))
            coupon = _to_float(cell(row, "Coupon (%)")) / 100.0
            holdings_rows.append(
                {
                    "cusip": synthetic_id(coupon, maturity),
                    "isin": "",
                    "name": name,
                    "par_value": _to_float(cell(row, "Par Value")),
                    "weight": _to_float(cell(row, "Weight (%)")),
                    "price": _to_float(cell(row, "Price")),
                    "maturity": maturity,
                    "coupon": coupon,
                }
            )
        elif "cash" in asset_class.lower() or "cash" in name.lower():
            # A cash row flows straight into the cash component.
            cash_component += _to_float(cell(row, "Market Value"))
        else:
            dropped.append(name)

    if cash_component == 0.0:
        console.log("No cash row found in iShares basket; cash component defaults to 0")

    return ParsedPCF(
        holdings=pd.DataFrame(holdings_rows),
        cash_component=cash_component,
        as_of=as_of,
        raw_row_count=len(rows),
        dropped=dropped,
    )


def parse_ishares(raw_df: pd.DataFrame) -> ParsedPCF:
    """Parse an iShares holdings CSV into the standardised basket shape.

    iShares holdings files carry columns such as Name, CUSIP, ISIN, Coupon (%),
    Maturity, Par Value, Weight (%), and Asset Class. We map those onto the
    standardised columns, sum cash-like rows into the cash component, and keep
    only Treasury rows.
    """

    def col(*candidates: str) -> str | None:
        for cand in candidates:
            for actual in raw_df.columns:
                if actual.strip().lower() == cand.lower():
                    return actual
        return None

    name_col = col("Name")
    isin_col = col("ISIN")
    coupon_col = col("Coupon (%)", "Coupon")
    maturity_col = col("Maturity")
    par_col = col("Par Value", "Notional Value", "Market Value")
    weight_col = col("Weight (%)", "Weight")
    price_col = col("Price")
    market_value_col = col("Market Value", "Market Value ($)")

    holdings_rows = []
    cash_component = 0.0
    dropped: list[str] = []

    for _, row in raw_df.iterrows():
        name = str(row.get(name_col, "")).strip() if name_col else ""
        if not name or name.lower() == "nan":
            continue
        par_value = _to_float(row.get(par_col)) if par_col else 0.0
        market_value = (
            _to_float(row.get(market_value_col)) if market_value_col else 0.0
        )
        if not _is_treasury(name):
            # Cash and cash-like rows accumulate into the cash component using
            # their market value; everything else is recorded as dropped.
            lower = name.lower()
            if any(m in lower for m in ("cash", "usd", "money market", "net other")):
                cash_component += market_value
            else:
                dropped.append(name)
            continue
        maturity = _parse_maturity(row.get(maturity_col)) if maturity_col else None
        coupon = _to_float(row.get(coupon_col)) / 100.0 if coupon_col else 0.0
        # No CUSIP is used as the key; the bond is identified by coupon+maturity.
        holdings_rows.append(
            {
                "cusip": synthetic_id(coupon, maturity),
                "isin": str(row.get(isin_col, "")).strip() if isin_col else "",
                "name": name,
                "par_value": par_value,
                "weight": _to_float(row.get(weight_col)) if weight_col else 0.0,
                "price": _to_float(row.get(price_col)) if price_col else 0.0,
                "maturity": maturity,
                "coupon": coupon,
            }
        )

    holdings = pd.DataFrame(holdings_rows)
    return ParsedPCF(
        holdings=holdings,
        cash_component=cash_component,
        raw_row_count=len(raw_df),
        dropped=dropped,
    )


def parse_vanguard(raw_df: pd.DataFrame) -> ParsedPCF:
    """Parse a Vanguard holdings file into the standardised basket shape.

    Vanguard publishes similar fields under different headers. The mapping is
    structurally identical to iShares; only the column names differ.
    """
    return _parse_generic(
        raw_df,
        name_candidates=("Name", "Security", "Holding"),
        cusip_candidates=("CUSIP", "SEDOL"),
        isin_candidates=("ISIN",),
        coupon_candidates=("Coupon", "Rate", "Coupon Rate"),
        maturity_candidates=("Maturity", "Maturity Date"),
        par_candidates=("Face Amount", "Par Value", "Shares"),
        weight_candidates=("% of fund", "Weight", "% of Net Assets"),
        market_value_candidates=("Market Value",),
    )


def parse_ssga(raw_df: pd.DataFrame) -> ParsedPCF:
    """Parse a State Street (SPDR) holdings file into the standardised shape.

    SPDR holdings spreadsheets carry yet another set of headers. Same mapping,
    different names.
    """
    return _parse_generic(
        raw_df,
        name_candidates=("Name", "Security Name"),
        cusip_candidates=("CUSIP",),
        isin_candidates=("ISIN",),
        coupon_candidates=("Coupon", "Coupon Rate"),
        maturity_candidates=("Maturity", "Maturity Date"),
        par_candidates=("Par Value", "Notional Value", "Shares Held"),
        weight_candidates=("Weight", "Weight (%)"),
        market_value_candidates=("Market Value", "Local Market Value"),
    )


def _parse_generic(raw_df: pd.DataFrame, **candidate_sets) -> ParsedPCF:
    """Shared column-mapping parser used by the Vanguard and SSGA parsers."""

    def col(candidates) -> str | None:
        for cand in candidates:
            for actual in raw_df.columns:
                if actual.strip().lower() == cand.lower():
                    return actual
        return None

    name_col = col(candidate_sets["name_candidates"])
    isin_col = col(candidate_sets["isin_candidates"])
    coupon_col = col(candidate_sets["coupon_candidates"])
    maturity_col = col(candidate_sets["maturity_candidates"])
    par_col = col(candidate_sets["par_candidates"])
    weight_col = col(candidate_sets["weight_candidates"])
    price_col = col(candidate_sets.get("price_candidates", ("Price",)))
    market_value_col = col(candidate_sets["market_value_candidates"])

    holdings_rows = []
    cash_component = 0.0
    dropped: list[str] = []

    for _, row in raw_df.iterrows():
        name = str(row.get(name_col, "")).strip() if name_col else ""
        if not name or name.lower() == "nan":
            continue
        market_value = (
            _to_float(row.get(market_value_col)) if market_value_col else 0.0
        )
        if not _is_treasury(name):
            lower = name.lower()
            if any(m in lower for m in ("cash", "usd", "money market", "net other")):
                cash_component += market_value
            else:
                dropped.append(name)
            continue
        maturity = _parse_maturity(row.get(maturity_col)) if maturity_col else None
        coupon = _to_float(row.get(coupon_col)) / 100.0 if coupon_col else 0.0
        # Identify the Treasury by coupon+maturity; no CUSIP is used as the key.
        holdings_rows.append(
            {
                "cusip": synthetic_id(coupon, maturity),
                "isin": str(row.get(isin_col, "")).strip() if isin_col else "",
                "name": name,
                "par_value": _to_float(row.get(par_col)) if par_col else 0.0,
                "weight": _to_float(row.get(weight_col)) if weight_col else 0.0,
                "price": _to_float(row.get(price_col)) if price_col else 0.0,
                "maturity": maturity,
                "coupon": coupon,
            }
        )

    return ParsedPCF(
        holdings=pd.DataFrame(holdings_rows),
        cash_component=cash_component,
        raw_row_count=len(raw_df),
        dropped=dropped,
    )


def _parse_maturity(value) -> date | None:
    """Parse a maturity cell into a date, tolerating several common formats."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(text).date()
    except Exception:  # noqa: BLE001
        return None


_PARSERS = {
    "ishares": parse_ishares,
    "vanguard": parse_vanguard,
    "ssga": parse_ssga,
}


def _local_holdings_path(ticker: str) -> Path | None:
    """Return a local holdings file for a ticker if one is present.

    Supports the common names data/{TICKER}.csv, data/{TICKER}_holdings.csv,
    data/{TICKER}.xls, and data/{TICKER}_fund.xls, so a manually downloaded basket
    can be dropped in without renaming. The .xls files are the real iShares
    SpreadsheetML format. Returns None when no local file exists.
    """
    ticker = ticker.upper()
    for name in (
        f"{ticker}.csv",
        f"{ticker}_holdings.csv",
        f"{ticker}.xls",
        f"{ticker}_fund.xls",
        f"{ticker}.xml",
    ):
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    return None


def load_pcf(ticker: str, provider: str, pcf_url: str | None = None) -> ParsedPCF:
    """Load and parse the PCF for a ticker, local file first then live fetch.

    A local holdings file under data/ (keyed by ticker) is used in preference to a
    live download, so a manually downloaded basket keeps the tool working for any
    provider when the live fetch is blocked. Otherwise the basket is fetched live
    with realistic browser headers.

    The format is detected from the content, not the extension: the real iShares
    file is a SpreadsheetML XML document served with a .xls extension, so it is
    parsed as XML; provider CSVs go through the CSV path, where a few metadata
    lines precede the holdings header. Returns a fully parsed, Treasury-only
    ParsedPCF with the cash component captured separately.
    """
    local = _local_holdings_path(ticker)
    if local is not None:
        console.log(f"Loading local holdings for {ticker} from {local}")
        raw_text = local.read_text(encoding="utf-8", errors="replace")
    else:
        url = pcf_url or build_pcf_url(ticker, provider)
        raw_text = _retry_download(url, provider, ticker)

    # SpreadsheetML (the real iShares ".xls"): parse as XML.
    if _looks_like_spreadsheetml(raw_text):
        parsed = parse_ishares_spreadsheetml(raw_text)
        console.log(
            f"Parsed iShares SpreadsheetML for {ticker}: {len(parsed.holdings)} "
            f"Treasury holdings as of {parsed.as_of}, cash "
            f"{parsed.cash_component:,.2f}, dropped {len(parsed.dropped)} rows"
        )
        return parsed

    # Defensive double-check. The live path already rejects HTML, but a local file
    # could also be a saved HTML page rather than the CSV.
    if _looks_like_html(raw_text):
        raise PCFDownloadError(
            f"Holdings source for {ticker} ({provider}) is an HTML page, not CSV."
        )

    # CSV path. The provider prepends fund metadata above the holdings table, so
    # we find the header line (one carrying a recognisable column) and parse from
    # there.
    lines = raw_text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        lowered = line.lower()
        if "cusip" in lowered or ("name" in lowered and "weight" in lowered):
            header_idx = i
            break
    cleaned = "\n".join(lines[header_idx:])
    raw_df = pd.read_csv(io.StringIO(cleaned), thousands=",")

    parser = _PARSERS.get(provider)
    if parser is None:
        raise ValueError(f"No parser for provider '{provider}'")
    parsed = parser(raw_df)
    console.log(
        f"Parsed PCF for {ticker}: {len(parsed.holdings)} Treasury holdings, "
        f"cash {parsed.cash_component:,.2f}, dropped {len(parsed.dropped)} rows"
    )
    return parsed

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
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
import requests
from rich.console import Console

from . import etf_config

console = Console()

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


def _retry_download(url: str, what: str) -> str:
    """Download a URL as text with the shared retry policy."""
    last_exc: Exception | None = None
    for attempt in range(1, etf_config.DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(
                url, timeout=30, headers={"User-Agent": "fi-etf-pricer"}
            )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
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
    cusip_col = col("CUSIP")
    isin_col = col("ISIN")
    coupon_col = col("Coupon (%)", "Coupon")
    maturity_col = col("Maturity")
    par_col = col("Par Value", "Notional Value", "Market Value")
    weight_col = col("Weight (%)", "Weight")
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
        holdings_rows.append(
            {
                "cusip": str(row.get(cusip_col, "")).strip() if cusip_col else "",
                "isin": str(row.get(isin_col, "")).strip() if isin_col else "",
                "name": name,
                "par_value": par_value,
                "weight": _to_float(row.get(weight_col)) if weight_col else 0.0,
                "maturity": _parse_maturity(row.get(maturity_col))
                if maturity_col
                else None,
                "coupon": _to_float(row.get(coupon_col)) / 100.0 if coupon_col else 0.0,
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
    cusip_col = col(candidate_sets["cusip_candidates"])
    isin_col = col(candidate_sets["isin_candidates"])
    coupon_col = col(candidate_sets["coupon_candidates"])
    maturity_col = col(candidate_sets["maturity_candidates"])
    par_col = col(candidate_sets["par_candidates"])
    weight_col = col(candidate_sets["weight_candidates"])
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
        holdings_rows.append(
            {
                "cusip": str(row.get(cusip_col, "")).strip() if cusip_col else "",
                "isin": str(row.get(isin_col, "")).strip() if isin_col else "",
                "name": name,
                "par_value": _to_float(row.get(par_col)) if par_col else 0.0,
                "weight": _to_float(row.get(weight_col)) if weight_col else 0.0,
                "maturity": _parse_maturity(row.get(maturity_col))
                if maturity_col
                else None,
                "coupon": _to_float(row.get(coupon_col)) / 100.0 if coupon_col else 0.0,
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


def load_pcf(ticker: str, provider: str, pcf_url: str | None = None) -> ParsedPCF:
    """Download and parse the PCF for a ticker.

    The iShares CSV carries a few metadata lines before the holdings header, so we
    locate the header row dynamically. Returns a fully parsed, Treasury-only
    ParsedPCF with the cash component captured separately.
    """
    url = pcf_url or build_pcf_url(ticker, provider)
    raw_text = _retry_download(url, f"PCF download for {ticker}")

    # Provider CDNs sometimes answer with an HTML bot-protection or consent page
    # instead of the CSV, especially from cloud IP ranges. Detect that fast and
    # raise a clear error rather than handing megabytes of HTML to the CSV parser.
    head = raw_text.lstrip()[:200].lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        raise ValueError(
            f"PCF endpoint for {ticker} returned an HTML page, not CSV. The "
            f"provider CDN is likely blocking this host; retry from a server "
            f"whose IP is not bot-filtered, or supply the basket via pcf_url."
        )

    # iShares prepends fund metadata above the holdings table. Find the header
    # line (the one that contains a recognisable column such as "CUSIP" or
    # "Name") and parse from there.
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

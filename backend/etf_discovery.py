"""ETF discovery and Treasury validation via Yahoo Finance.

When a user adds a ticker we must (a) confirm it really is a US Treasury ETF,
because this tool can only price Treasuries, and (b) work out which issuer runs
it so we know where to fetch the basket from. Both come from yfinance metadata.

All yfinance calls go through the shared three-attempt retry with a five second
wait, since Yahoo occasionally rate limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console

from . import etf_config

console = Console()

# Map the fundFamily string Yahoo returns onto our internal provider key. The
# provider key drives PCF URL construction and parser selection downstream.
PROVIDER_BY_FAMILY: dict[str, str] = {
    "ishares": "ishares",
    "vanguard": "vanguard",
    "spdr": "ssga",
    "state street": "ssga",
    "ssga": "ssga",
}


class NotATreasuryETF(Exception):
    """Raised when a ticker is not a US Treasury ETF this tool can price."""


class UnknownProvider(Exception):
    """Raised when the issuer is not one we have a PCF pattern for."""


@dataclass
class DiscoveryResult:
    """What we learn about a ticker from Yahoo Finance."""

    ticker: str
    name: str
    category: str
    provider: str
    fund_family: str
    current_price: float | None


def _retry_yf(fn, what: str):
    """Run a yfinance call with the shared retry policy.

    Three attempts, five second wait, per specification. yfinance raises a broad
    range of exceptions, so we catch broadly and re-raise the last one.
    """
    last_exc: Exception | None = None
    for attempt in range(1, etf_config.DOWNLOAD_RETRIES + 1):
        try:
            return fn()
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


def fetch_info(ticker: str) -> dict:
    """Fetch the raw yfinance info dictionary for a ticker, with retries."""
    import yfinance as yf

    return _retry_yf(lambda: yf.Ticker(ticker).info, f"yfinance info for {ticker}")


def detect_provider(fund_family: str) -> str:
    """Map a Yahoo fundFamily string to an internal provider key.

    iShares -> ishares, Vanguard -> vanguard, SPDR/State Street -> ssga. Anything
    else raises UnknownProvider so the caller can ask for a manual PCF URL.
    """
    family_lower = (fund_family or "").lower()
    for needle, provider in PROVIDER_BY_FAMILY.items():
        if needle in family_lower:
            return provider
    raise UnknownProvider(
        f"Unrecognised fund family '{fund_family}'. Provide a manual PCF URL."
    )


def discover(ticker: str) -> DiscoveryResult:
    """Validate that a ticker is a US Treasury ETF and detect its provider.

    Validation rules, applied in order against the Yahoo info dictionary:

    1. quoteType must equal "ETF", otherwise reject (it is not a fund).
    2. category must contain "Treasury" or "Government", otherwise reject with a
       message that this tool prices Treasuries only. This is the credit gate:
       corporate, municipal, mortgage, and multi-sector funds need spread or
       vendor pricing the tool does not have.
    3. fundFamily must map to a known provider so we can locate the basket.

    Returns a DiscoveryResult with the name, category, provider, and current
    price. Raises NotATreasuryETF or UnknownProvider on failure.
    """
    ticker = ticker.upper().strip()
    info = fetch_info(ticker)

    quote_type = info.get("quoteType", "")
    if quote_type != "ETF":
        raise NotATreasuryETF(
            f"{ticker} is not an ETF (quoteType={quote_type!r})."
        )

    category = info.get("category", "") or ""
    if "treasury" not in category.lower() and "government" not in category.lower():
        raise NotATreasuryETF(
            f"{ticker} is not a US Treasury ETF. This tool prices Treasuries only."
        )

    fund_family = info.get("fundFamily", "") or ""
    provider = detect_provider(fund_family)

    current_price = info.get("regularMarketPrice") or info.get(
        "regularMarketPreviousClose"
    )

    return DiscoveryResult(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName") or ticker,
        category=category,
        provider=provider,
        fund_family=fund_family,
        current_price=current_price,
    )


def fetch_closing_price(ticker: str, valuation_date) -> float | None:
    """Fetch the ETF closing market price for the valuation date.

    Uses the daily history and returns the close on or before the valuation date.
    This is the live-traded price side of the premium/discount comparison.
    """
    import yfinance as yf

    def _fetch() -> float | None:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        hist = hist[hist.index.date <= valuation_date]
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])

    return _retry_yf(_fetch, f"yfinance close for {ticker}")


def fetch_official_nav(ticker: str) -> float | None:
    """Fetch the issuer official published NAV from yfinance as a fallback.

    The provider NAV history file is preferred when available; this navPrice is
    the fallback used for the NAV tracking check.
    """
    info = fetch_info(ticker)
    return info.get("navPrice")

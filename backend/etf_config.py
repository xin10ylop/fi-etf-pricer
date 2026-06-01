"""Static configuration for the Treasury ETF pricer.

This module holds modelled assumptions and per-ETF static data. None of the
values here are observed market quotes. In particular the bid/ask spreads are
modelled assumptions used to turn the single fair (mid) price produced by the
curve into a bid/mid/ask. They are documented as such throughout so a reviewer
never mistakes them for real quotes.

All money figures are in US dollars and all bond prices are quoted per 100 face
value, following standard US Treasury convention.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Curve download
# ---------------------------------------------------------------------------

# Standard tenors published on the US Treasury Daily Par Yield Curve, expressed
# in years. These are the maturities the bootstrap walks through in order.
CURVE_TENORS_YEARS: dict[str, float] = {
    "1M": 1.0 / 12.0,
    "2M": 2.0 / 12.0,
    "3M": 3.0 / 12.0,
    "4M": 4.0 / 12.0,
    "6M": 6.0 / 12.0,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# Mapping from FRED series id to tenor in years, used only when the FRED
# alternative source is selected via configuration. FRED requires a free API
# key in the environment (FRED_API_KEY).
FRED_SERIES_TENORS: dict[str, float] = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 3.0 / 12.0,
    "DGS6MO": 6.0 / 12.0,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

# Number of coupon payments per year for US Treasury notes and bonds. The whole
# curve bootstrap and discounting machinery assumes this convention.
COUPONS_PER_YEAR: int = 2

# Network retry policy applied to every external download (curve, PCF,
# yfinance). Three attempts, five seconds between them, per the specification.
DOWNLOAD_RETRIES: int = 3
DOWNLOAD_RETRY_WAIT_SECONDS: int = 5

# ---------------------------------------------------------------------------
# Modelled bid/ask assumptions (NOT observed quotes)
# ---------------------------------------------------------------------------

# Half the modelled bid/ask spread for a Treasury, in price points per 100 face.
# The curve produces a single fair mid; we widen it symmetrically by this amount
# to model bid and ask. Treasuries are extremely liquid so this is intentionally
# small. This is a modelled assumption, not a real quote.
TREASURY_HALF_SPREAD_PRICE: float = 0.03

# Modelled half-spread on the ETF itself, in basis points of the ETF price. Used
# to estimate the trading cost of the ETF leg of a creation/redemption.
ETF_HALF_SPREAD_BPS: float = 1.0

# ---------------------------------------------------------------------------
# Per-ETF static routing data
# ---------------------------------------------------------------------------

# Creation unit sizes and authorised participant fees are published by the
# issuer. These defaults cover the starting watchlist. Anything discovered at
# runtime that is not present here falls back to DEFAULT_CREATION_UNIT_SHARES
# and DEFAULT_CREATION_FEE.
DEFAULT_CREATION_UNIT_SHARES: int = 50_000
DEFAULT_CREATION_FEE: float = 500.0

ETF_STATIC: dict[str, dict] = {
    "SHY": {"creation_unit_shares": 100_000, "creation_fee": 500.0},
    "IEI": {"creation_unit_shares": 100_000, "creation_fee": 500.0},
    "IEF": {"creation_unit_shares": 100_000, "creation_fee": 500.0},
    "TLT": {"creation_unit_shares": 100_000, "creation_fee": 500.0},
    "GOVT": {"creation_unit_shares": 100_000, "creation_fee": 500.0},
}

# The starting watchlist. The specification calls for building and testing IEF
# end to end first, then adding the rest.
DEFAULT_WATCHLIST: list[str] = ["IEF", "SHY", "IEI", "TLT", "GOVT"]

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

# Daily repricing time, expressed in US Eastern time. The Treasury par curve for
# the day is published in the afternoon, so we reprice after the close.
SCHEDULE_TIME_ET: str = "17:30"
SCHEDULE_TIMEZONE: str = "America/New_York"

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

DATABASE_PATH: str = "etf_pricer.db"


def creation_unit_shares(ticker: str) -> int:
    """Return the creation unit size for a ticker, falling back to the default.

    The creation unit is the fixed number of ETF shares exchanged in a single
    creation or redemption with the issuer. NAV per share is the basket value
    divided by this number.
    """
    return ETF_STATIC.get(ticker.upper(), {}).get(
        "creation_unit_shares", DEFAULT_CREATION_UNIT_SHARES
    )


def creation_fee(ticker: str) -> float:
    """Return the authorised participant creation fee in US dollars.

    This is a flat fee charged by the issuer per creation or redemption and is
    one component of the total arbitrage cost.
    """
    return ETF_STATIC.get(ticker.upper(), {}).get(
        "creation_fee", DEFAULT_CREATION_FEE
    )

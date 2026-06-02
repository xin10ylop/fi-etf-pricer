"""Repricing engine: NAV, premium/discount, costs, and the trade signal.

This module ties the pieces together for one ETF on one valuation date:

  curve -> price each Treasury -> store marks -> aggregate basket in SQL
        -> NAV per share -> compare to the ETF close -> net off costs -> signal

CONTEMPORANEOUS MARKS
The signal compares the ETF closing price against an NAV built from the SAME
day's closing Treasury curve. Both sides are end-of-day and carry the curve date,
so we are never comparing a live ETF price to a stale NAV. Avoiding that stale-NAV
trap is the whole design point: most apparent bond-ETF premiums are an artefact of
the underlying being marked once a day while the ETF trades continuously.

NAV USES DIRTY PRICES
An ETF NAV includes accrued interest, so every NAV figure here is built from dirty
prices, never clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from rich.console import Console

from . import database, etf_config
from .curve import TreasuryCurve, build_curve
from .etf_discovery import fetch_closing_price, fetch_official_nav
from .pcf_loader import DATA_DIR, load_pcf
from .pricing import Bond
from .price_provider import CurvePriceProvider, PriceProvider

console = Console()


@dataclass
class RepricingResult:
    """The full output of repricing one ETF, ready to store and serve."""

    ticker: str
    curve_date: str
    valuation_ts: str
    source: str
    basket_dirty_mid: float
    basket_dirty_bid: float
    basket_dirty_ask: float
    cash_component: float
    nav_per_share: float
    official_nav: float | None
    nav_tracking_bps: float | None
    etf_price: float | None
    premium_bps: float | None
    creation_fee: float
    bond_spread_cost: float
    etf_trading_cost: float
    total_costs: float
    breakeven_bps: float
    signal: str
    net_edge_usd: float
    extras: dict = field(default_factory=dict)


def _build_bonds(basket: list[dict]) -> list[Bond]:
    """Turn stored basket rows into Bond objects for pricing.

    Rows missing a maturity or coupon cannot be priced off the curve and are
    skipped with a warning, since a Treasury without a maturity is unusable.
    """
    bonds: list[Bond] = []
    for row in basket:
        maturity = row.get("maturity")
        if isinstance(maturity, str):
            maturity = date.fromisoformat(maturity)
        if maturity is None or row.get("coupon") is None:
            console.log(f"[yellow]Skipping {row.get('cusip')}: missing maturity/coupon[/yellow]")
            continue
        bonds.append(
            Bond(
                cusip=row["cusip"],
                coupon=float(row["coupon"]),
                maturity=maturity,
                name=row.get("name", "") or "",
            )
        )
    return bonds


def refresh_basket(
    ticker: str, db_path: str = etf_config.DATABASE_PATH
) -> float:
    """Download and store the latest PCF for a ticker.

    Downloads the basket live, stores each Treasury's reference data and the
    holding rows, and records the cash component on the routing row. Returns the
    cash component in US dollars. This is run before each repricing so the basket
    and cash are current for the valuation date.
    """
    routing = database.get_routing(ticker, db_path)
    if routing is None:
        raise ValueError(f"{ticker} is not in the routing table; add it first.")
    parsed = load_pcf(ticker, routing["provider"], routing.get("pcf_url"))
    holdings = []
    for _, row in parsed.holdings.iterrows():
        database.upsert_bond(
            cusip=row["cusip"],
            name=row["name"],
            coupon=float(row["coupon"]),
            maturity=row["maturity"],
            db_path=db_path,
        )
        holdings.append(
            {
                "cusip": row["cusip"],
                "par_value": float(row["par_value"]),
                "weight": float(row["weight"]),
            }
        )
    database.replace_basket(ticker, holdings, db_path)
    database.update_cash_component(ticker, parsed.cash_component, db_path)
    database.update_basket_as_of(ticker, parsed.as_of, db_path)
    database.update_shares_outstanding(ticker, parsed.shares_outstanding, db_path)
    return parsed.cash_component


def save_uploaded_basket(
    ticker: str, content: bytes, db_path: str = etf_config.DATABASE_PATH
) -> dict:
    """Persist a manually uploaded holdings file and load it into the basket.

    The uploaded bytes are written into the data/ folder under the ticker, where
    the loader looks first, so every later repricing uses the manual basket until
    it is replaced. The format (SpreadsheetML .xls or CSV) is detected from the
    content, not the file name, then refresh_basket parses and stores it. Returns
    the holding count and the basket as-of date.
    """
    routing = database.get_routing(ticker, db_path)
    if routing is None:
        # Default watchlist tickers are seeded without a routing row (routing is
        # otherwise created on discovery). A manual upload supplies the basket
        # directly, so we create a minimal routing row for any watchlisted ticker.
        if ticker.upper() not in database.get_watchlist(db_path):
            raise ValueError(
                f"{ticker} is not on the watchlist; add it before uploading a basket."
            )
        database.upsert_routing(
            ticker=ticker.upper(),
            name=ticker.upper(),
            provider="ishares",
            pcf_url="",
            creation_unit_shares=etf_config.creation_unit_shares(ticker),
            creation_fee=etf_config.creation_fee(ticker),
            category="Treasury",
            db_path=db_path,
        )

    text_head = content[:512].decode("utf-8", errors="replace").lstrip().lower()
    if "<html" in text_head or text_head.startswith("<!doctype html"):
        raise ValueError("Uploaded file is an HTML page, not a holdings export.")
    # Choose a filename the loader recognises; the parser routes by content.
    is_spreadsheetml = (
        "<ss:workbook" in text_head
        or "urn:schemas-microsoft-com:office:spreadsheet" in text_head
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any prior local file for this ticker so the new upload wins cleanly.
    for existing in DATA_DIR.glob(f"{ticker.upper()}.*"):
        existing.unlink()
    target = DATA_DIR / (f"{ticker.upper()}.xls" if is_spreadsheetml else f"{ticker.upper()}.csv")
    target.write_bytes(content)

    cash = refresh_basket(ticker, db_path)
    routing = database.get_routing(ticker, db_path) or {}
    return {
        "ticker": ticker.upper(),
        "holdings": len(database.get_basket(ticker, db_path)),
        "cash_component": cash,
        "basket_as_of": routing.get("basket_as_of"),
    }


def compute_signal(
    etf_price: float | None,
    nav_per_share: float,
    premium_bps: float | None,
    breakeven_bps: float,
) -> str:
    """Classify the trade signal from the premium against the breakeven band.

    premium > +breakeven  -> CREATE (ETF rich vs basket; create and sell)
    premium < -breakeven  -> REDEEM (ETF cheap vs basket; buy and redeem)
    otherwise             -> NO TRADE (inside the cost band)
    """
    if etf_price is None or premium_bps is None:
        return "NO TRADE"
    if premium_bps > breakeven_bps:
        return "CREATE"
    if premium_bps < -breakeven_bps:
        return "REDEEM"
    return "NO TRADE"


def reprice(
    ticker: str,
    valuation_date: date | None = None,
    curve: TreasuryCurve | None = None,
    provider: PriceProvider | None = None,
    refresh_pcf: bool = True,
    db_path: str = etf_config.DATABASE_PATH,
) -> RepricingResult:
    """Reprice one ETF end to end and persist the result.

    Steps:
      1. Build (or accept) the bootstrapped curve for the valuation date and store
         its points.
      2. Load the stored basket and price every Treasury via the price provider
         (the curve provider by default). Persist the marks.
      3. Aggregate the basket dirty bid/mid/ask in SQL and add the cash component.
      4. NAV per share = (basket dirty mid + cash) / creation unit shares.
      5. premium_bps = (etf_price / nav - 1) * 10000.
      6. Cost stack: creation fee + bond half-spread cost + ETF trading cost.
         The bond half-spread cost is side-aware: a CREATE buys the basket so it
         pays the ask side, a REDEEM sells so it earns/pays the bid side.
      7. breakeven_bps = total costs / NAV value of one creation unit, in bps.
      8. Signal from premium against the breakeven band; net edge in dollars.

    Returns a RepricingResult and writes marks and the signal to SQLite.
    """
    valuation_date = valuation_date or date.today()
    valuation_ts = datetime.now(timezone.utc).isoformat()

    routing = database.get_routing(ticker, db_path)
    if routing is None:
        raise ValueError(f"{ticker} is not in the routing table; add it first.")

    # 0. Refresh the basket and cash component from the live PCF.
    if refresh_pcf:
        refresh_basket(ticker, db_path)
        routing = database.get_routing(ticker, db_path)

    # 1. Curve.
    if curve is None:
        curve = build_curve(valuation_date)
    database.store_curve_points(curve.curve_date, curve.points, db_path)

    # 2. Price the basket.
    basket = database.get_basket(ticker, db_path)
    if not basket:
        raise ValueError(f"No basket stored for {ticker}; run a PCF load first.")
    bonds = _build_bonds(basket)
    if provider is None:
        provider = CurvePriceProvider(curve)
    priced = provider.price_bonds(bonds, valuation_date)
    database.store_marks(ticker, curve.curve_date, valuation_ts, priced, db_path)

    # 3. Aggregate the basket in SQL and add cash. The cash component is held on
    # the routing row and was refreshed from the latest PCF above.
    agg = database.aggregate_basket_nav(ticker, curve.curve_date, db_path)
    cash_component = float(routing.get("cash_component", 0.0) or 0.0)
    basket_dirty_mid = agg["basket_dirty_mid"]
    basket_dirty_bid = agg["basket_dirty_bid"]
    basket_dirty_ask = agg["basket_dirty_ask"]

    # 4. NAV per share.
    # The iShares basket lists the whole fund's holdings, so the basket value must
    # be divided by the fund's shares outstanding, not the creation unit size. The
    # creation unit size is kept below for the cost and signal calculations only.
    creation_unit_shares = int(
        routing.get("creation_unit_shares") or etf_config.creation_unit_shares(ticker)
    )
    shares_outstanding = routing.get("shares_outstanding")
    if not shares_outstanding or float(shares_outstanding) <= 0:
        raise ValueError(
            f"Cannot compute NAV for {ticker}: shares outstanding is missing or "
            f"zero. Upload a current iShares basket file, which carries shares "
            f"outstanding in its metadata."
        )
    shares_outstanding = float(shares_outstanding)
    nav_per_share = (basket_dirty_mid + cash_component) / shares_outstanding

    # 5. Premium against the ETF close (contemporaneous, same curve date).
    etf_price = fetch_closing_price_safe(ticker, valuation_date)
    premium_bps = (
        (etf_price / nav_per_share - 1.0) * 10000.0
        if etf_price and nav_per_share
        else None
    )

    # 6. Cost stack.
    creation_fee = float(routing.get("creation_fee") or etf_config.creation_fee(ticker))
    create_spread_cost = basket_dirty_ask - basket_dirty_mid
    redeem_spread_cost = basket_dirty_mid - basket_dirty_bid
    # Default the reported bond spread cost to the create side; refined below once
    # the signal direction is known.
    bond_half_spread_cost = create_spread_cost
    etf_trading_cost = (
        (etf_price or nav_per_share)
        * creation_unit_shares
        * etf_config.ETF_HALF_SPREAD_BPS
        / 10000.0
    )

    # 7. Breakeven in bps of one creation unit of NAV.
    creation_unit_value = nav_per_share * creation_unit_shares
    total_costs = creation_fee + bond_half_spread_cost + etf_trading_cost
    breakeven_bps = (
        total_costs / creation_unit_value * 10000.0 if creation_unit_value else 0.0
    )

    # 8. Signal, then refine the side-aware spread cost and recompute the band.
    signal = compute_signal(etf_price, nav_per_share, premium_bps, breakeven_bps)
    if signal == "REDEEM":
        bond_half_spread_cost = redeem_spread_cost
        total_costs = creation_fee + bond_half_spread_cost + etf_trading_cost
        breakeven_bps = (
            total_costs / creation_unit_value * 10000.0 if creation_unit_value else 0.0
        )
        signal = compute_signal(etf_price, nav_per_share, premium_bps, breakeven_bps)

    net_edge_usd = (
        abs(premium_bps / 10000.0) * creation_unit_value - total_costs
        if premium_bps is not None
        else 0.0
    )

    # NAV tracking check against the issuer official NAV.
    official_nav = fetch_official_nav_safe(ticker)
    nav_tracking_bps = (
        (nav_per_share / official_nav - 1.0) * 10000.0
        if official_nav
        else None
    )

    result = RepricingResult(
        ticker=ticker.upper(),
        curve_date=curve.curve_date.isoformat(),
        valuation_ts=valuation_ts,
        source=getattr(provider, "source_label", "curve"),
        basket_dirty_mid=basket_dirty_mid,
        basket_dirty_bid=basket_dirty_bid,
        basket_dirty_ask=basket_dirty_ask,
        cash_component=cash_component,
        nav_per_share=nav_per_share,
        official_nav=official_nav,
        nav_tracking_bps=nav_tracking_bps,
        etf_price=etf_price,
        premium_bps=premium_bps,
        creation_fee=creation_fee,
        bond_spread_cost=bond_half_spread_cost,
        etf_trading_cost=etf_trading_cost,
        total_costs=total_costs,
        breakeven_bps=breakeven_bps,
        signal=signal,
        net_edge_usd=net_edge_usd,
    )

    database.store_signal(
        {
            "curve_date": result.curve_date,
            "valuation_ts": result.valuation_ts,
            "ticker": result.ticker,
            "etf_price": result.etf_price,
            "nav_per_share": result.nav_per_share,
            "official_nav": result.official_nav,
            "nav_tracking_bps": result.nav_tracking_bps,
            "premium_bps": result.premium_bps,
            "bond_spread_cost": result.bond_spread_cost,
            "creation_fee": result.creation_fee,
            "breakeven_bps": result.breakeven_bps,
            "signal": result.signal,
            "net_edge_usd": result.net_edge_usd,
        },
        db_path,
    )

    console.log(
        f"[green]{ticker}[/green] NAV {nav_per_share:.4f} "
        f"price {etf_price} premium {premium_bps} bps signal {signal} "
        f"track {nav_tracking_bps} bps"
    )
    return result


def fetch_closing_price_safe(ticker: str, valuation_date: date) -> float | None:
    """Fetch the ETF close, returning None on failure rather than raising.

    A repricing run should still produce NAV and tracking figures even if the live
    ETF price is briefly unavailable; the signal simply becomes NO TRADE.
    """
    try:
        return fetch_closing_price(ticker, valuation_date)
    except Exception as exc:  # noqa: BLE001
        console.log(f"[yellow]ETF close fetch failed for {ticker}: {exc}[/yellow]")
        return None


def fetch_official_nav_safe(ticker: str) -> float | None:
    """Fetch the official NAV, returning None on failure rather than raising."""
    try:
        return fetch_official_nav(ticker)
    except Exception as exc:  # noqa: BLE001
        console.log(f"[yellow]Official NAV fetch failed for {ticker}: {exc}[/yellow]")
        return None

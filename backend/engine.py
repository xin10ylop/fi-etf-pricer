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
    # NAV bridge and reconciliation (added for the audit).
    curve_nav: float | None = None
    vendor_nav: float | None = None
    vendor_timing_bps: float | None = None
    curve_vs_vendor_bps: float | None = None
    premium_vs_curve_bps: float | None = None
    premium_vs_official_bps: float | None = None
    confidence_band_bps: float | None = None
    effective_threshold: float | None = None
    basket_as_of: str | None = None
    etf_price_date: str | None = None
    official_nav_date: str | None = None
    treasury_dirty_value: float | None = None
    shares_outstanding: float | None = None
    mean_abs_diff_bps: float | None = None
    max_abs_diff_bps: float | None = None
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
                "vendor_price": float(row["price"]) if row.get("price") else None,
            }
        )
    database.replace_basket(ticker, holdings, db_path)
    database.update_cash_component(ticker, parsed.cash_component, db_path)
    database.update_basket_as_of(ticker, parsed.as_of, db_path)
    database.update_shares_outstanding(ticker, parsed.shares_outstanding, db_path)
    database.update_vendor_metadata(
        ticker, parsed.vendor_mv_total, parsed.official_nav, parsed.official_nav_date,
        db_path,
    )
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


def _fmt(value: float | None, digits: int = 2) -> str:
    """Format an optional number for logging, showing 'NA' when None."""
    return "NA" if value is None else f"{value:.{digits}f}"


def compute_signal(
    etf_price: float | None,
    premium_bps: float | None,
    threshold_bps: float,
) -> str:
    """Classify the trade signal from the premium against an effective threshold.

    The threshold is the breakeven cost plus the confidence band (the engine's own
    pricing error), so a trade must clear BOTH cost and our pricing uncertainty.

    premium > +threshold  -> CREATE (ETF rich vs basket; create and sell)
    premium < -threshold  -> REDEEM (ETF cheap vs basket; buy and redeem)
    otherwise             -> NO TRADE
    """
    if etf_price is None or premium_bps is None:
        return "NO TRADE"
    if premium_bps > threshold_bps:
        return "CREATE"
    if premium_bps < -threshold_bps:
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
    valuation_date_explicit = valuation_date is not None
    valuation_date = valuation_date or date.today()
    valuation_ts = datetime.now(timezone.utc).isoformat()

    routing = database.get_routing(ticker, db_path)
    if routing is None:
        raise ValueError(f"{ticker} is not in the routing table; add it first.")

    # 0. Refresh the basket and cash component from the live PCF.
    if refresh_pcf:
        refresh_basket(ticker, db_path)
        routing = database.get_routing(ticker, db_path)

    # ISSUE 3: align all inputs to the basket as-of date. This is an end-of-day
    # tool, so the curve, the ETF close, and the official NAV are all taken as of
    # the date the basket was struck, unless the caller pinned an explicit date.
    basket_as_of_str = routing.get("basket_as_of")
    basket_as_of = date.fromisoformat(basket_as_of_str) if basket_as_of_str else None
    if not valuation_date_explicit and basket_as_of is not None:
        valuation_date = basket_as_of

    # 1. Curve dated to the basket as-of date (build_curve uses the most recent
    # published curve on or before that date).
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
    treasury_dirty_value = basket_dirty_mid
    curve_nav = (treasury_dirty_value + cash_component) / shares_outstanding
    nav_per_share = curve_nav

    # ISSUE 7: NAV bridge. Decompose tracking into a vendor timing/basis effect and
    # a curve-vs-vendor pricing effect.
    #   official_nav -> +vendor_timing -> vendor_nav -> +curve_vs_vendor -> curve_nav
    vendor_mv_total = routing.get("vendor_mv_total")
    vendor_nav = (
        float(vendor_mv_total) / shares_outstanding if vendor_mv_total else None
    )
    official_nav = routing.get("official_nav") or fetch_official_nav_safe(ticker)
    official_nav = float(official_nav) if official_nav else None
    official_nav_date = routing.get("official_nav_date") or (
        valuation_date.isoformat()
    )
    vendor_timing_bps = (
        (vendor_nav / official_nav - 1.0) * 10000.0
        if vendor_nav and official_nav
        else None
    )
    curve_vs_vendor_bps = (
        (curve_nav / vendor_nav - 1.0) * 10000.0 if vendor_nav else None
    )
    nav_tracking_bps = (
        (curve_nav / official_nav - 1.0) * 10000.0 if official_nav else None
    )

    # 5. ETF closing price ON the basket as-of date (end-of-day, not intraday).
    etf_price = fetch_closing_price_safe(ticker, valuation_date)
    etf_price_date = valuation_date.isoformat()

    # ISSUE 5: premium against BOTH the curve NAV and the official NAV.
    premium_vs_curve_bps = (
        (etf_price / curve_nav - 1.0) * 10000.0 if etf_price and curve_nav else None
    )
    premium_vs_official_bps = (
        (etf_price / official_nav - 1.0) * 10000.0
        if etf_price and official_nav
        else None
    )

    # 6. Cost stack. ISSUE 2: the bond half-spread is a whole-fund figure, so scale
    # it to a single creation unit before comparing to per-creation-unit value.
    creation_fee = float(routing.get("creation_fee") or etf_config.creation_fee(ticker))
    creation_unit_weight = creation_unit_shares / shares_outstanding
    create_spread_cost = (basket_dirty_ask - basket_dirty_mid) * creation_unit_weight
    redeem_spread_cost = (basket_dirty_mid - basket_dirty_bid) * creation_unit_weight
    bond_half_spread_cost = create_spread_cost  # refined once direction is known
    etf_trading_cost = (
        (etf_price or curve_nav)
        * creation_unit_shares
        * etf_config.ETF_HALF_SPREAD_BPS
        / 10000.0
    )

    # 7. Breakeven in bps of one creation unit of NAV.
    creation_unit_value = curve_nav * creation_unit_shares
    total_costs = creation_fee + bond_half_spread_cost + etf_trading_cost
    breakeven_bps = (
        total_costs / creation_unit_value * 10000.0 if creation_unit_value else 0.0
    )

    # ISSUE 5: the signal must clear cost AND the engine's own pricing error. The
    # confidence band is the absolute tracking error against the official NAV.
    confidence_band_bps = abs(nav_tracking_bps) if nav_tracking_bps is not None else 0.0
    effective_threshold = breakeven_bps + confidence_band_bps

    # 8. Signal from premium-vs-curve against the effective threshold. Refine the
    # side-aware spread cost for the chosen direction and recompute the band.
    signal = compute_signal(etf_price, premium_vs_curve_bps, effective_threshold)
    if signal == "REDEEM":
        bond_half_spread_cost = redeem_spread_cost
        total_costs = creation_fee + bond_half_spread_cost + etf_trading_cost
        breakeven_bps = (
            total_costs / creation_unit_value * 10000.0 if creation_unit_value else 0.0
        )
        effective_threshold = breakeven_bps + confidence_band_bps
        signal = compute_signal(etf_price, premium_vs_curve_bps, effective_threshold)

    net_edge_usd = (
        abs(premium_vs_curve_bps / 10000.0) * creation_unit_value - total_costs
        if premium_vs_curve_bps is not None
        else 0.0
    )

    # ISSUE 6: per-bond reconciliation of curve clean vs the file vendor price.
    vendor_by_cusip = {row["cusip"]: row.get("vendor_price") for row in basket}
    abs_diffs: list[float] = []
    for pb in priced:
        vp = vendor_by_cusip.get(pb.cusip)
        if vp:
            abs_diffs.append(abs((pb.clean_mid / float(vp) - 1.0) * 10000.0))
    mean_abs_diff_bps = sum(abs_diffs) / len(abs_diffs) if abs_diffs else None
    max_abs_diff_bps = max(abs_diffs) if abs_diffs else None

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
        premium_bps=premium_vs_curve_bps,
        creation_fee=creation_fee,
        bond_spread_cost=bond_half_spread_cost,
        etf_trading_cost=etf_trading_cost,
        total_costs=total_costs,
        breakeven_bps=breakeven_bps,
        signal=signal,
        net_edge_usd=net_edge_usd,
        curve_nav=curve_nav,
        vendor_nav=vendor_nav,
        vendor_timing_bps=vendor_timing_bps,
        curve_vs_vendor_bps=curve_vs_vendor_bps,
        premium_vs_curve_bps=premium_vs_curve_bps,
        premium_vs_official_bps=premium_vs_official_bps,
        confidence_band_bps=confidence_band_bps,
        effective_threshold=effective_threshold,
        basket_as_of=basket_as_of.isoformat() if basket_as_of else None,
        etf_price_date=etf_price_date,
        official_nav_date=official_nav_date,
        treasury_dirty_value=treasury_dirty_value,
        shares_outstanding=shares_outstanding,
        mean_abs_diff_bps=mean_abs_diff_bps,
        max_abs_diff_bps=max_abs_diff_bps,
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
            "premium_vs_curve_bps": premium_vs_curve_bps,
            "premium_vs_official_bps": premium_vs_official_bps,
            "curve_nav": curve_nav,
            "vendor_nav": vendor_nav,
            "vendor_timing_bps": vendor_timing_bps,
            "curve_vs_vendor_bps": curve_vs_vendor_bps,
            "confidence_band_bps": confidence_band_bps,
            "effective_threshold": effective_threshold,
            "basket_as_of": result.basket_as_of,
            "etf_price_date": etf_price_date,
            "official_nav_date": official_nav_date,
            "treasury_dirty_value": treasury_dirty_value,
            "cash_component": cash_component,
            "shares_outstanding": shares_outstanding,
            "mean_abs_diff_bps": mean_abs_diff_bps,
            "max_abs_diff_bps": max_abs_diff_bps,
        },
        db_path,
    )

    # ISSUE 4: reconciliation log so any future discrepancy is diagnosable.
    console.rule(f"[amber]Reconciliation {ticker} as of {basket_as_of}[/amber]")
    console.log(
        f"curve_date={curve.curve_date} treasury_dirty={treasury_dirty_value:,.2f} "
        f"cash={cash_component:,.2f} shares_out={shares_outstanding:,.0f}"
    )
    console.log(
        f"curve_nav={curve_nav:.6f} vendor_nav="
        f"{vendor_nav if vendor_nav is None else round(vendor_nav,6)} "
        f"official_nav={official_nav} tracking_bps="
        f"{nav_tracking_bps if nav_tracking_bps is None else round(nav_tracking_bps,2)}"
    )
    console.log(
        f"bridge: official ->{_fmt(vendor_timing_bps)}bps-> vendor "
        f"->{_fmt(curve_vs_vendor_bps)}bps-> curve | "
        f"per-bond mean|diff|={_fmt(mean_abs_diff_bps)} max={_fmt(max_abs_diff_bps)}"
    )
    console.log(
        f"etf_price={etf_price} prem_vs_curve={_fmt(premium_vs_curve_bps)} "
        f"prem_vs_official={_fmt(premium_vs_official_bps)} "
        f"breakeven={breakeven_bps:.2f} conf_band={confidence_band_bps:.2f} "
        f"eff_threshold={effective_threshold:.2f} signal={signal}"
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

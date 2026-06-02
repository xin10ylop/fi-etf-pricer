"""FastAPI application: REST endpoints, the dashboard, and the daily scheduler.

The whole app runs from one address and one port: FastAPI serves the dashboard
at the root path and the REST API under /api, and the dashboard calls the API
with same-origin relative paths. CORS is still open so the dashboard can also be
hosted elsewhere (for example Netlify) against this same backend. The scheduler
runs in a background thread, repricing the whole watchlist once a day after the
Treasury curve is published, while FastAPI keeps serving requests.

Every response that carries a price or NAV also carries the curve date and a
valuation timestamp, preserving the contemporaneous end-of-day design end to end.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime
from pathlib import Path

import schedule
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rich.console import Console

from . import database, etf_config
from .engine import refresh_basket, reprice, save_uploaded_basket
from .etf_discovery import (
    NotATreasuryETF,
    UnknownProvider,
    discover,
)
from .pcf_loader import build_pcf_url

console = Console()

# The single-file dashboard lives alongside the backend in the repo. We resolve
# its path relative to this module so it serves correctly regardless of the
# working directory uvicorn is launched from.
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

app = FastAPI(
    title="Treasury ETF Pricer",
    description=(
        "Fair value and creation/redemption arbitrage signals for US Treasury "
        "ETFs, priced from the bootstrapped Treasury curve."
    ),
    version="1.0.0",
)

# Open CORS so the dashboard can call this API from any origin. When served from
# this same app the calls are same-origin and CORS is not exercised, but leaving
# it open keeps a separately hosted dashboard working too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the single-file dashboard at the root path.

    Visiting the server address shows the dashboard directly, so the whole app
    runs from one address and one port: the page and the API it calls share the
    same origin.
    """
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found")
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class TickerBody(BaseModel):
    """Request body carrying a single ticker."""

    ticker: str


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    """Initialise the database and seed the default watchlist on startup."""
    database.init_db()
    for ticker in etf_config.DEFAULT_WATCHLIST:
        database.add_to_watchlist(ticker)
    start_scheduler()


# ---------------------------------------------------------------------------
# Discovery and watchlist
# ---------------------------------------------------------------------------


@app.get("/api/discover")
def api_discover(ticker: str) -> dict:
    """Validate that a ticker is a US Treasury ETF and report what we found.

    Returns the name, provider, category, and current price. Rejects anything that
    is not a Treasury ETF with a clear message, since the tool prices Treasuries
    only.
    """
    try:
        result = discover(ticker)
    except NotATreasuryETF as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except UnknownProvider as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Discovery failed: {exc}")
    return {
        "ticker": result.ticker,
        "name": result.name,
        "provider": result.provider,
        "category": result.category,
        "current_price": result.current_price,
    }


@app.post("/api/watchlist/add")
def api_watchlist_add(body: TickerBody) -> dict:
    """Discover, validate, do a test PCF download, and add a ticker.

    The test download proves we can actually fetch and parse the basket before we
    commit the ticker to the watchlist, so a fund we cannot price never gets added.
    """
    ticker = body.ticker.upper().strip()
    try:
        result = discover(ticker)
    except (NotATreasuryETF, UnknownProvider) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Discovery failed: {exc}")

    try:
        pcf_url = build_pcf_url(ticker, result.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    database.upsert_routing(
        ticker=ticker,
        name=result.name,
        provider=result.provider,
        pcf_url=pcf_url,
        creation_unit_shares=etf_config.creation_unit_shares(ticker),
        creation_fee=etf_config.creation_fee(ticker),
        category=result.category,
    )

    try:
        n_holdings = _test_pcf_download(ticker)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"PCF test download failed for {ticker}: {exc}",
        )

    database.add_to_watchlist(ticker)
    return {
        "ticker": ticker,
        "name": result.name,
        "provider": result.provider,
        "holdings": n_holdings,
        "status": "added",
    }


def _test_pcf_download(ticker: str) -> int:
    """Refresh the basket once and return the Treasury holding count."""
    refresh_basket(ticker)
    return len(database.get_basket(ticker))


@app.delete("/api/watchlist/remove")
def api_watchlist_remove(body: TickerBody) -> dict:
    """Remove a ticker from the watchlist."""
    database.remove_from_watchlist(body.ticker)
    return {"ticker": body.ticker.upper(), "status": "removed"}


@app.get("/api/watchlist")
def api_watchlist() -> dict:
    """Return every watchlist ETF with its latest signal summary."""
    summary = []
    for ticker in database.get_watchlist():
        routing = database.get_routing(ticker) or {}
        sig = database.latest_signal(ticker)
        summary.append(
            {
                "ticker": ticker,
                "name": routing.get("name", ticker),
                "signal": sig["signal"] if sig else None,
                "premium_bps": sig["premium_bps"] if sig else None,
                "nav_per_share": sig["nav_per_share"] if sig else None,
                "etf_price": sig["etf_price"] if sig else None,
                "nav_tracking_bps": sig["nav_tracking_bps"] if sig else None,
                "net_edge_usd": sig["net_edge_usd"] if sig else None,
                "last_updated": sig["valuation_ts"] if sig else None,
                "basket_as_of": routing.get("basket_as_of"),
            }
        )
    return {"watchlist": summary}


# ---------------------------------------------------------------------------
# Signal detail
# ---------------------------------------------------------------------------


@app.get("/api/signals/{ticker}")
def api_signal_detail(ticker: str) -> dict:
    """Return full detail for one ETF: summary, costs, top holdings, history.

    The top holdings carry clean, accrued, and dirty marks so the dashboard can
    show the pricing detail that makes the NAV auditable.
    """
    ticker = ticker.upper()
    routing = database.get_routing(ticker)
    if routing is None:
        raise HTTPException(status_code=404, detail=f"{ticker} not found")
    sig = database.latest_signal(ticker)
    if sig is None:
        raise HTTPException(status_code=404, detail=f"No signal yet for {ticker}")

    curve_date = date.fromisoformat(sig["curve_date"])
    marks = database.latest_marks(ticker, curve_date)

    def diff_bps(m: dict) -> float | None:
        """Per-bond reconciliation: curve clean vs the file vendor clean price."""
        vp = m.get("vendor_price")
        if vp and m.get("clean_mid"):
            return (m["clean_mid"] / vp - 1.0) * 10000.0
        return None

    top_holdings = [
        {
            "name": m.get("name"),
            "cusip": m["cusip"],
            "coupon": m.get("coupon"),
            "maturity": m.get("maturity"),
            "weight": m.get("weight"),
            "clean_mid": m["clean_mid"],
            "accrued_interest": m["accrued_interest"],
            "dirty_mid": m["dirty_mid"],
            "vendor_price": m.get("vendor_price"),
            "diff_bps": diff_bps(m),
        }
        for m in marks[:10]
    ]

    history = [
        {
            "curve_date": h["curve_date"],
            "premium_bps": h.get("premium_vs_curve_bps") or h["premium_bps"],
        }
        for h in database.signal_history(ticker, days=30)
    ]

    return {
        "summary": {
            "ticker": ticker,
            "name": routing.get("name"),
            "curve_date": sig["curve_date"],
            "last_updated": sig["valuation_ts"],
            "basket_as_of": sig.get("basket_as_of") or routing.get("basket_as_of"),
            "etf_price_date": sig.get("etf_price_date"),
            "official_nav_date": sig.get("official_nav_date"),
            "source": "curve",
            "nav_per_share": sig["nav_per_share"],
            "curve_nav": sig.get("curve_nav"),
            "official_nav": sig["official_nav"],
            "nav_tracking_bps": sig["nav_tracking_bps"],
            "etf_price": sig["etf_price"],
            "premium_bps": sig["premium_bps"],
            "premium_vs_curve_bps": sig.get("premium_vs_curve_bps"),
            "premium_vs_official_bps": sig.get("premium_vs_official_bps"),
            "confidence_band_bps": sig.get("confidence_band_bps"),
            "effective_threshold": sig.get("effective_threshold"),
            "signal": sig["signal"],
            "net_edge_usd": sig["net_edge_usd"],
        },
        "costs": {
            "creation_fee": sig["creation_fee"],
            "bond_spread_cost": sig["bond_spread_cost"],
            "breakeven_bps": sig["breakeven_bps"],
            "confidence_band_bps": sig.get("confidence_band_bps"),
            "effective_threshold": sig.get("effective_threshold"),
        },
        "bridge": {
            "official_nav": sig["official_nav"],
            "vendor_timing_bps": sig.get("vendor_timing_bps"),
            "vendor_nav": sig.get("vendor_nav"),
            "curve_vs_vendor_bps": sig.get("curve_vs_vendor_bps"),
            "curve_nav": sig.get("curve_nav"),
        },
        "reconciliation": {
            "treasury_dirty_value": sig.get("treasury_dirty_value"),
            "cash_component": sig.get("cash_component"),
            "shares_outstanding": sig.get("shares_outstanding"),
            "curve_nav": sig.get("curve_nav"),
            "official_nav": sig["official_nav"],
            "nav_tracking_bps": sig["nav_tracking_bps"],
            "mean_abs_diff_bps": sig.get("mean_abs_diff_bps"),
            "max_abs_diff_bps": sig.get("max_abs_diff_bps"),
        },
        "top_holdings": top_holdings,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Repricing triggers
# ---------------------------------------------------------------------------


@app.post("/api/upload-basket")
async def api_upload_basket(ticker: str, file: UploadFile = File(...)) -> dict:
    """Accept a manually uploaded holdings file and load it as the basket.

    The file (an iShares SpreadsheetML .xls or a provider CSV) is posted as
    multipart form data with the ticker as a query parameter. It is persisted and
    parsed into the basket, and the basket as-of date is recorded. The user is
    then prompted to refresh (reprice). Returns the holding count and as-of date.
    """
    content = await file.read()
    try:
        result = save_uploaded_basket(ticker, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Basket upload failed: {exc}")
    return {**result, "status": "uploaded"}


@app.post("/api/run")
def api_run(ticker: str) -> dict:
    """Trigger an immediate repricing for one ETF and return the new signal."""
    try:
        result = reprice(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Repricing failed: {exc}")
    return _result_to_dict(result)


@app.post("/api/run-all")
def api_run_all() -> dict:
    """Trigger an immediate repricing for every watchlist ETF."""
    results = run_all_watchlist()
    return {"repriced": results}


def _result_to_dict(result) -> dict:
    """Flatten a RepricingResult for JSON transport."""
    return {
        "ticker": result.ticker,
        "curve_date": result.curve_date,
        "valuation_ts": result.valuation_ts,
        "source": result.source,
        "nav_per_share": result.nav_per_share,
        "official_nav": result.official_nav,
        "nav_tracking_bps": result.nav_tracking_bps,
        "etf_price": result.etf_price,
        "premium_bps": result.premium_bps,
        "signal": result.signal,
        "breakeven_bps": result.breakeven_bps,
        "net_edge_usd": result.net_edge_usd,
        "bond_spread_cost": result.bond_spread_cost,
        "creation_fee": result.creation_fee,
        "etf_trading_cost": result.etf_trading_cost,
        "total_costs": result.total_costs,
        "basket_dirty_mid": result.basket_dirty_mid,
        "cash_component": result.cash_component,
        "curve_nav": result.curve_nav,
        "vendor_nav": result.vendor_nav,
        "vendor_timing_bps": result.vendor_timing_bps,
        "curve_vs_vendor_bps": result.curve_vs_vendor_bps,
        "premium_vs_curve_bps": result.premium_vs_curve_bps,
        "premium_vs_official_bps": result.premium_vs_official_bps,
        "confidence_band_bps": result.confidence_band_bps,
        "effective_threshold": result.effective_threshold,
        "basket_as_of": result.basket_as_of,
        "mean_abs_diff_bps": result.mean_abs_diff_bps,
        "max_abs_diff_bps": result.max_abs_diff_bps,
    }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def run_all_watchlist() -> list[dict]:
    """Reprice every ticker on the watchlist, logging and skipping failures.

    One bad ticker must not abort the daily run, so each ticker is repriced inside
    its own try/except.
    """
    results = []
    for ticker in database.get_watchlist():
        try:
            result = reprice(ticker)
            results.append(_result_to_dict(result))
        except Exception as exc:  # noqa: BLE001
            console.log(f"[red]Repricing {ticker} failed: {exc}[/red]")
            results.append({"ticker": ticker, "error": str(exc)})
    return results


def _scheduler_loop() -> None:
    """Background loop that runs pending scheduled jobs once a minute.

    The daily job is registered for the configured Eastern-time close. The loop
    sleeps between checks so it never blocks the FastAPI event loop.
    """
    while True:
        schedule.run_pending()
        time.sleep(30)


def start_scheduler() -> None:
    """Register the daily repricing job and start the background scheduler.

    The schedule library runs in local server time; deployments should set the
    server timezone to US Eastern (or run in UTC and adjust the configured time)
    so the job fires after the Treasury curve is published for the day.
    """
    schedule.clear()
    schedule.every().day.at(etf_config.SCHEDULE_TIME_ET).do(_scheduled_repricing)
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    console.log(
        f"Scheduler started: daily repricing at {etf_config.SCHEDULE_TIME_ET} "
        f"({etf_config.SCHEDULE_TIMEZONE})"
    )


def _scheduled_repricing() -> None:
    """The job the scheduler fires: reprice the whole watchlist."""
    console.rule(f"[amber]Scheduled repricing {datetime.now().isoformat()}[/amber]")
    run_all_watchlist()


@app.get("/api/health")
def api_health() -> dict:
    """Liveness probe reporting the server time and watchlist size."""
    return {
        "status": "ok",
        "server_time": datetime.now().isoformat(),
        "watchlist_size": len(database.get_watchlist()),
    }

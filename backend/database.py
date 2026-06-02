"""SQLite storage for routing, baskets, curve points, marks, and signals.

Every mark and signal is stored with both a valuation timestamp and the curve
date, so the contemporaneity of the comparison (ETF close versus an NAV built
from the same day's curve) is always visible and auditable.

A uniqueness constraint on (ticker, cusip, curve_date) in the marks table is the
single most important integrity guard here: it stops the basket NAV sum from
fanning out if a join ever produced duplicate mark rows. The NAV aggregation is
done in SQL so the sum is computed where the data lives.

Note on the bond key: the column is named "cusip" for continuity, but the iShares
holdings file carries no CUSIP, so the value stored is the coupon+maturity
synthetic identifier (for example "UST_4.250_2034-11-15") built in pcf_loader. A
US Treasury is uniquely identified by its coupon and maturity, so this is a valid
unique key, and the schema and the NAV aggregation SQL are unchanged.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date

from rich.console import Console

from . import etf_config

console = Console()


SCHEMA = """
CREATE TABLE IF NOT EXISTS routing (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    provider TEXT,
    pcf_url TEXT,
    creation_unit_shares INTEGER,
    creation_fee REAL,
    category TEXT,
    cash_component REAL DEFAULT 0.0,
    basket_as_of TEXT
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS bonds (
    cusip TEXT PRIMARY KEY,
    name TEXT,
    coupon REAL,
    maturity TEXT
);

CREATE TABLE IF NOT EXISTS basket_holdings (
    ticker TEXT,
    cusip TEXT,
    par_value REAL,
    weight REAL,
    PRIMARY KEY (ticker, cusip)
);

CREATE TABLE IF NOT EXISTS curve_points (
    curve_date TEXT,
    tenor_years REAL,
    par_yield REAL,
    zero_rate REAL,
    PRIMARY KEY (curve_date, tenor_years)
);

CREATE TABLE IF NOT EXISTS marks (
    ticker TEXT,
    cusip TEXT,
    curve_date TEXT,
    valuation_ts TEXT,
    source TEXT,
    clean_bid REAL,
    clean_mid REAL,
    clean_ask REAL,
    accrued_interest REAL,
    dirty_bid REAL,
    dirty_mid REAL,
    dirty_ask REAL,
    PRIMARY KEY (ticker, cusip, curve_date)
);

CREATE TABLE IF NOT EXISTS signals (
    curve_date TEXT,
    valuation_ts TEXT,
    ticker TEXT,
    etf_price REAL,
    nav_per_share REAL,
    official_nav REAL,
    nav_tracking_bps REAL,
    premium_bps REAL,
    bond_spread_cost REAL,
    creation_fee REAL,
    breakeven_bps REAL,
    signal TEXT,
    net_edge_usd REAL,
    PRIMARY KEY (ticker, curve_date)
);
"""


@contextmanager
def connect(db_path: str = etf_config.DATABASE_PATH):
    """Open a SQLite connection with row access by column name.

    Foreign keys and a row factory are enabled. Used as a context manager so the
    connection is always committed and closed.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = etf_config.DATABASE_PATH) -> None:
    """Create all tables if they do not exist and run lightweight migrations."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
    console.log("Database schema ready")


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    CREATE TABLE IF NOT EXISTS never alters an existing table, so a database made
    before basket_as_of existed would lack that column. We add it idempotently.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(routing)")}
    if "basket_as_of" not in existing:
        conn.execute("ALTER TABLE routing ADD COLUMN basket_as_of TEXT")
    if "cash_component" not in existing:
        conn.execute(
            "ALTER TABLE routing ADD COLUMN cash_component REAL DEFAULT 0.0"
        )


# ---------------------------------------------------------------------------
# Routing and watchlist
# ---------------------------------------------------------------------------


def upsert_routing(
    ticker: str,
    name: str,
    provider: str,
    pcf_url: str,
    creation_unit_shares: int,
    creation_fee: float,
    category: str,
    db_path: str = etf_config.DATABASE_PATH,
) -> None:
    """Insert or replace the routing record for a ticker."""
    with connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO routing
               (ticker, name, provider, pcf_url, creation_unit_shares,
                creation_fee, category)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker.upper(),
                name,
                provider,
                pcf_url,
                creation_unit_shares,
                creation_fee,
                category,
            ),
        )


def update_cash_component(
    ticker: str, cash_component: float, db_path: str = etf_config.DATABASE_PATH
) -> None:
    """Update the stored cash component for a ticker from the latest PCF.

    The cash component is the dollar cash held in one creation unit. It flows
    straight into NAV without being priced, and it changes day to day, so it is
    refreshed every time the PCF is loaded.
    """
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE routing SET cash_component = ? WHERE ticker = ?",
            (cash_component, ticker.upper()),
        )


def update_basket_as_of(
    ticker: str, as_of: date | None, db_path: str = etf_config.DATABASE_PATH
) -> None:
    """Record the basket as-of date for a ticker from the latest holdings file.

    The as-of date is the "Fund Holdings as of" date published in the basket file.
    Storing it lets the dashboard show how fresh the holdings are. None clears it,
    which surfaces as "No basket loaded" in the UI.
    """
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE routing SET basket_as_of = ? WHERE ticker = ?",
            (as_of.isoformat() if as_of else None, ticker.upper()),
        )


def get_routing(ticker: str, db_path: str = etf_config.DATABASE_PATH) -> dict | None:
    """Return the routing record for a ticker, or None if not present."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM routing WHERE ticker = ?", (ticker.upper(),)
        ).fetchone()
        return dict(row) if row else None


def add_to_watchlist(ticker: str, db_path: str = etf_config.DATABASE_PATH) -> None:
    """Add a ticker to the watchlist (idempotent)."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (ticker.upper(),)
        )


def remove_from_watchlist(ticker: str, db_path: str = etf_config.DATABASE_PATH) -> None:
    """Remove a ticker from the watchlist."""
    with connect(db_path) as conn:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))


def get_watchlist(db_path: str = etf_config.DATABASE_PATH) -> list[str]:
    """Return all tickers currently on the watchlist."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall()
        return [r["ticker"] for r in rows]


# ---------------------------------------------------------------------------
# Bonds, baskets, curve points
# ---------------------------------------------------------------------------


def upsert_bond(
    cusip: str,
    name: str,
    coupon: float,
    maturity: date | None,
    db_path: str = etf_config.DATABASE_PATH,
) -> None:
    """Insert or replace a bond's static reference data."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO bonds (cusip, name, coupon, maturity) "
            "VALUES (?, ?, ?, ?)",
            (cusip, name, coupon, maturity.isoformat() if maturity else None),
        )


def replace_basket(
    ticker: str, holdings: list[dict], db_path: str = etf_config.DATABASE_PATH
) -> None:
    """Replace the stored basket for a ticker with a fresh set of holdings.

    The old basket is cleared first so a shrinking basket never leaves stale
    holdings behind. Each holding carries cusip, par_value, and weight.
    """
    with connect(db_path) as conn:
        conn.execute("DELETE FROM basket_holdings WHERE ticker = ?", (ticker.upper(),))
        conn.executemany(
            "INSERT OR REPLACE INTO basket_holdings "
            "(ticker, cusip, par_value, weight) VALUES (?, ?, ?, ?)",
            [
                (ticker.upper(), h["cusip"], h["par_value"], h["weight"])
                for h in holdings
            ],
        )


def get_basket(ticker: str, db_path: str = etf_config.DATABASE_PATH) -> list[dict]:
    """Return the stored basket holdings for a ticker joined with bond data."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT h.cusip, h.par_value, h.weight,
                      b.name, b.coupon, b.maturity
               FROM basket_holdings h
               LEFT JOIN bonds b ON b.cusip = h.cusip
               WHERE h.ticker = ?
               ORDER BY h.weight DESC""",
            (ticker.upper(),),
        ).fetchall()
        return [dict(r) for r in rows]


def store_curve_points(
    curve_date: date, points: list, db_path: str = etf_config.DATABASE_PATH
) -> None:
    """Store the bootstrapped curve points (par yield and zero rate per tenor)."""
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO curve_points "
            "(curve_date, tenor_years, par_yield, zero_rate) VALUES (?, ?, ?, ?)",
            [
                (curve_date.isoformat(), p.tenor_years, p.par_yield, p.zero_rate)
                for p in points
            ],
        )


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------


def store_marks(
    ticker: str,
    curve_date: date,
    valuation_ts: str,
    priced_bonds: list,
    db_path: str = etf_config.DATABASE_PATH,
) -> None:
    """Store priced marks for a basket.

    The (ticker, cusip, curve_date) primary key guarantees one mark per bond per
    day, so the NAV aggregation join cannot fan out. INSERT OR REPLACE makes a
    re-run for the same day overwrite cleanly.
    """
    with connect(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO marks
               (ticker, cusip, curve_date, valuation_ts, source,
                clean_bid, clean_mid, clean_ask, accrued_interest,
                dirty_bid, dirty_mid, dirty_ask)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ticker.upper(),
                    pb.cusip,
                    curve_date.isoformat(),
                    valuation_ts,
                    pb.source,
                    pb.clean_bid,
                    pb.clean_mid,
                    pb.clean_ask,
                    pb.accrued_interest,
                    pb.dirty_bid,
                    pb.dirty_mid,
                    pb.dirty_ask,
                )
                for pb in priced_bonds
            ],
        )


def aggregate_basket_nav(
    ticker: str, curve_date: date, db_path: str = etf_config.DATABASE_PATH
) -> dict:
    """Aggregate the basket dirty value in SQL.

    The NAV is built from DIRTY prices because an ETF NAV includes accrued
    interest. The sum is computed in SQL, where the basket and the marks live, so
    no Python-side row multiplication can creep in. Returns the dirty bid, mid,
    and ask basket totals in US dollars.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT
                   SUM(h.par_value * m.dirty_mid / 100.0) AS basket_dirty_mid,
                   SUM(h.par_value * m.dirty_bid / 100.0) AS basket_dirty_bid,
                   SUM(h.par_value * m.dirty_ask / 100.0) AS basket_dirty_ask
               FROM basket_holdings h
               JOIN marks m ON m.cusip = h.cusip AND m.ticker = h.ticker
               WHERE h.ticker = ? AND m.curve_date = ?""",
            (ticker.upper(), curve_date.isoformat()),
        ).fetchone()
        return {
            "basket_dirty_mid": row["basket_dirty_mid"] or 0.0,
            "basket_dirty_bid": row["basket_dirty_bid"] or 0.0,
            "basket_dirty_ask": row["basket_dirty_ask"] or 0.0,
        }


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def store_signal(signal: dict, db_path: str = etf_config.DATABASE_PATH) -> None:
    """Store a computed signal row, one per (ticker, curve_date)."""
    with connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signals
               (curve_date, valuation_ts, ticker, etf_price, nav_per_share,
                official_nav, nav_tracking_bps, premium_bps, bond_spread_cost,
                creation_fee, breakeven_bps, signal, net_edge_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal["curve_date"],
                signal["valuation_ts"],
                signal["ticker"],
                signal["etf_price"],
                signal["nav_per_share"],
                signal["official_nav"],
                signal["nav_tracking_bps"],
                signal["premium_bps"],
                signal["bond_spread_cost"],
                signal["creation_fee"],
                signal["breakeven_bps"],
                signal["signal"],
                signal["net_edge_usd"],
            ),
        )


def latest_signal(ticker: str, db_path: str = etf_config.DATABASE_PATH) -> dict | None:
    """Return the most recent signal for a ticker by curve date."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM signals WHERE ticker = ? ORDER BY curve_date DESC LIMIT 1",
            (ticker.upper(),),
        ).fetchone()
        return dict(row) if row else None


def signal_history(
    ticker: str, days: int = 30, db_path: str = etf_config.DATABASE_PATH
) -> list[dict]:
    """Return the last N signals for a ticker, oldest first, for charting."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM signals WHERE ticker = ? ORDER BY curve_date DESC LIMIT ?",
            (ticker.upper(), days),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def latest_marks(
    ticker: str, curve_date: date, db_path: str = etf_config.DATABASE_PATH
) -> list[dict]:
    """Return the stored marks for a basket on a curve date, joined with bonds."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT m.*, b.name, b.coupon, b.maturity, h.par_value, h.weight
               FROM marks m
               JOIN basket_holdings h ON h.cusip = m.cusip AND h.ticker = m.ticker
               LEFT JOIN bonds b ON b.cusip = m.cusip
               WHERE m.ticker = ? AND m.curve_date = ?
               ORDER BY h.weight DESC""",
            (ticker.upper(), curve_date.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

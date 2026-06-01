"""US Treasury yield curve engine.

This is the heart of the project. It does three things, in order:

1. DOWNLOAD the US Treasury Daily Par Yield Curve Rates for a valuation date.
   These are free, public, and require no API key. An alternative FRED source is
   available behind a configuration flag for days the Treasury feed is awkward.

2. BOOTSTRAP a zero-coupon (spot) curve from the par yields. Par yields are the
   coupon rates that make a bond price to exactly 100; they are NOT discount
   rates. We recover the discount rates by solving sequentially, shortest tenor
   first, so that a freshly issued par bond at each tenor reprices to 100.

3. INTERPOLATE the zero curve to return a discount factor for a cash flow at any
   time in years.

COMPOUNDING CONVENTION
Everything in this module uses SEMIANNUAL compounding, matching the semiannual
coupon convention of US Treasury notes and bonds. A zero rate z at time t (in
years) maps to a discount factor by

    discount_factor(t) = (1 + z / 2) ** (-2 * t)

and inversely

    z(t) = 2 * (discount_factor(t) ** (-1 / (2 * t)) - 1)

This convention is applied consistently in the bootstrap, the interpolation, and
all downstream bond pricing.

THE BOOTSTRAP MATHS
A par bond of maturity T with semiannual coupons pays a coupon of (c / 2) per 100
face on each of n = 2T coupon dates t_1, t_2, ..., t_n, and repays 100 at t_n.
By definition of a par yield it prices to 100:

    100 = (c / 2 * 100) * sum_{i=1}^{n} DF(t_i)  +  100 * DF(t_n)

If we already know DF(t_1) ... DF(t_{n-1}) from shorter tenors, the only unknown
is DF(t_n), and we solve it directly:

    DF(t_n) = (100 - (c / 2 * 100) * sum_{i=1}^{n-1} DF(t_i))
              / (c / 2 * 100 + 100)

We walk the semiannual time grid from 0.5 years outward, applying this formula at
each step. Par yields are only published at a handful of tenors, so we first
linearly interpolate the published par yields onto the full semiannual grid
before bootstrapping.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import requests
from rich.console import Console

from . import etf_config

console = Console()

# The Treasury publishes the daily par yield curve as CSV via this endpoint. The
# field_tdr_date_value_month query selects the month; we then filter to the exact
# valuation date in the returned rows.
TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)

# Mapping from the Treasury CSV column headers to tenors in years. The CSV uses
# headers like "1 Mo", "3 Mo", "1 Yr", "10 Yr".
TREASURY_COLUMN_TENORS: dict[str, float] = {
    "1 Mo": 1.0 / 12.0,
    "2 Mo": 2.0 / 12.0,
    "3 Mo": 3.0 / 12.0,
    "4 Mo": 4.0 / 12.0,
    "6 Mo": 6.0 / 12.0,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}


@dataclass
class CurvePoint:
    """A single point on the curve at a standard tenor.

    par_yield and zero_rate are stored as decimals (0.045 means 4.5%).
    """

    tenor_years: float
    par_yield: float
    zero_rate: float


@dataclass
class TreasuryCurve:
    """A bootstrapped Treasury zero curve for one valuation date.

    Holds the fine semiannual grid of times and bootstrapped zero rates, plus
    the par yields interpolated onto that grid. The discount_factor method is the
    single interface the rest of the system uses to discount cash flows.
    """

    curve_date: date
    grid_times: np.ndarray  # semiannual grid, years
    grid_zero_rates: np.ndarray  # bootstrapped zero rates on the grid, decimals
    grid_par_yields: np.ndarray  # par yields interpolated onto the grid, decimals
    points: list[CurvePoint] = field(default_factory=list)

    def discount_factor(self, t_years: float) -> float:
        """Return the discount factor for a cash flow at t_years from valuation.

        The zero rate is linearly interpolated on the bootstrapped grid (linear
        on zero rates is the chosen convention; it is simple, monotone-friendly,
        and adequate for the smoothly shaped Treasury curve). For times before
        the first grid point or beyond the last, the nearest grid zero rate is
        held flat. The interpolated zero rate is then converted to a discount
        factor under the semiannual compounding convention.
        """
        if t_years <= 0.0:
            return 1.0
        z = float(
            np.interp(
                t_years,
                self.grid_times,
                self.grid_zero_rates,
                left=self.grid_zero_rates[0],
                right=self.grid_zero_rates[-1],
            )
        )
        return (1.0 + z / etf_config.COUPONS_PER_YEAR) ** (
            -etf_config.COUPONS_PER_YEAR * t_years
        )

    def zero_rate(self, t_years: float) -> float:
        """Return the interpolated zero (spot) rate at t_years, as a decimal."""
        if t_years <= 0.0:
            return float(self.grid_zero_rates[0])
        return float(
            np.interp(
                t_years,
                self.grid_times,
                self.grid_zero_rates,
                left=self.grid_zero_rates[0],
                right=self.grid_zero_rates[-1],
            )
        )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _retry(fn, what: str):
    """Run a download function with the configured retry policy.

    Three attempts with a five second wait between them, per specification. The
    last exception is re-raised if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(1, etf_config.DOWNLOAD_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last_exc = exc
            console.log(
                f"[yellow]{what} attempt {attempt}/"
                f"{etf_config.DOWNLOAD_RETRIES} failed: {exc}[/yellow]"
            )
            if attempt < etf_config.DOWNLOAD_RETRIES:
                time.sleep(etf_config.DOWNLOAD_RETRY_WAIT_SECONDS)
    assert last_exc is not None
    raise last_exc


def download_treasury_par_yields(valuation_date: date) -> dict[float, float]:
    """Download the Treasury par yield curve for the valuation date.

    Returns a mapping of tenor in years to par yield as a decimal. The Treasury
    CSV holds one row per business day in the year; we pick the row matching the
    valuation date. Retries are applied per the configured policy.
    """

    def _fetch() -> dict[float, float]:
        url = TREASURY_CSV_URL.format(year=valuation_date.year)
        resp = requests.get(url, timeout=30, headers={"User-Agent": "fi-etf-pricer"})
        resp.raise_for_status()
        frame = pd.read_csv(io.StringIO(resp.text))
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
        row = frame[frame["Date"] == valuation_date]
        if row.empty:
            # Fall back to the most recent available row on or before the date.
            earlier = frame[frame["Date"] <= valuation_date]
            if earlier.empty:
                raise ValueError(
                    f"No Treasury par curve available on or before {valuation_date}"
                )
            row = earlier.sort_values("Date").iloc[[-1]]
        record = row.iloc[0]
        par_yields: dict[float, float] = {}
        for column, tenor in TREASURY_COLUMN_TENORS.items():
            if column in record and pd.notna(record[column]):
                par_yields[tenor] = float(record[column]) / 100.0
        if not par_yields:
            raise ValueError("Treasury CSV returned no usable tenors")
        return par_yields

    return _retry(_fetch, "Treasury par curve download")


def download_fred_par_yields(valuation_date: date, api_key: str) -> dict[float, float]:
    """Download par yields from FRED as the alternative source.

    FRED requires a free API key. Each constant-maturity Treasury series is
    queried for its observation on the valuation date. Returns tenor-in-years to
    par yield as a decimal. Used only when the FRED source is explicitly
    selected.
    """

    def _fetch() -> dict[float, float]:
        par_yields: dict[float, float] = {}
        for series, tenor in etf_config.FRED_SERIES_TENORS.items():
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series}&api_key={api_key}&file_type=json"
                f"&observation_start={valuation_date}&observation_end={valuation_date}"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
            for obs in observations:
                value = obs.get("value")
                if value not in (None, ".", ""):
                    par_yields[tenor] = float(value) / 100.0
        if not par_yields:
            raise ValueError(f"FRED returned no usable tenors for {valuation_date}")
        return par_yields

    return _retry(_fetch, "FRED par curve download")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_zero_curve(
    par_yields: dict[float, float], curve_date: date
) -> TreasuryCurve:
    """Bootstrap a zero curve from published par yields.

    Steps, all under the semiannual convention documented at module level:

    1. Linearly interpolate the published par yields onto a complete semiannual
       grid (0.5, 1.0, 1.5, ... up to the longest published tenor). Coupons fall
       on this grid, so we need a par yield at every grid point.
    2. Walk the grid shortest first. At each tenor the par bond prices to 100, and
       every earlier discount factor is already known, so we solve the one
       remaining discount factor directly with the bootstrap formula.
    3. Convert each discount factor to a zero rate for storage and interpolation.

    Returns a TreasuryCurve carrying the full grid plus CurvePoints at the
    standard published tenors.
    """
    tenors = sorted(par_yields)
    max_tenor = tenors[-1]

    # Semiannual grid from 0.5 years out to the longest tenor.
    step = 1.0 / etf_config.COUPONS_PER_YEAR
    n_steps = int(round(max_tenor / step))
    grid_times = np.array([step * (i + 1) for i in range(n_steps)])

    # Linear interpolation of par yields onto the grid. Below the shortest and
    # above the longest published tenor the nearest yield is held flat.
    known_tenors = np.array(tenors)
    known_yields = np.array([par_yields[t] for t in tenors])
    grid_par = np.interp(
        grid_times, known_tenors, known_yields,
        left=known_yields[0], right=known_yields[-1],
    )

    # Bootstrap discount factors sequentially.
    discount_factors = np.zeros(n_steps)
    running_sum = 0.0
    for k in range(n_steps):
        coupon = grid_par[k] / etf_config.COUPONS_PER_YEAR * 100.0
        df_k = (100.0 - coupon * running_sum) / (coupon + 100.0)
        discount_factors[k] = df_k
        running_sum += df_k

    # Convert discount factors to semiannually compounded zero rates.
    grid_zero = etf_config.COUPONS_PER_YEAR * (
        discount_factors ** (-1.0 / (etf_config.COUPONS_PER_YEAR * grid_times)) - 1.0
    )

    curve = TreasuryCurve(
        curve_date=curve_date,
        grid_times=grid_times,
        grid_zero_rates=grid_zero,
        grid_par_yields=grid_par,
    )

    # Build CurvePoints at the standard published tenors for storage. The zero
    # rate at each is read back from the interpolated grid.
    for tenor in tenors:
        curve.points.append(
            CurvePoint(
                tenor_years=tenor,
                par_yield=par_yields[tenor],
                zero_rate=curve.zero_rate(tenor),
            )
        )

    return curve


def build_curve(valuation_date: date, source: str = "treasury") -> TreasuryCurve:
    """Download par yields and bootstrap the zero curve for the valuation date.

    source is "treasury" (default, free, no key) or "fred" (requires FRED_API_KEY
    in the environment). The result is a fully bootstrapped TreasuryCurve.
    """
    if source == "fred":
        import os

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED source selected but FRED_API_KEY is not set")
        par_yields = download_fred_par_yields(valuation_date, api_key)
    else:
        par_yields = download_treasury_par_yields(valuation_date)
    return bootstrap_zero_curve(par_yields, valuation_date)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def reprice_par_bond(curve: TreasuryCurve, tenor_years: float, par_yield: float) -> float:
    """Reprice a par bond at a tenor off the bootstrapped curve.

    A correct bootstrap must reprice each input par bond back to 100. This is the
    headline sanity check for the curve engine. Returns the computed dirty price
    per 100 face, which should be very close to 100.
    """
    step = 1.0 / etf_config.COUPONS_PER_YEAR
    n = int(round(tenor_years / step))
    coupon = par_yield / etf_config.COUPONS_PER_YEAR * 100.0
    price = 0.0
    for i in range(1, n + 1):
        t = step * i
        price += coupon * curve.discount_factor(t)
    price += 100.0 * curve.discount_factor(step * n)
    return price


if __name__ == "__main__":
    # Offline sanity check using a synthetic upward sloping par curve. This proves
    # the bootstrap is self consistent: each par bond must reprice to 100.
    sample_par = {
        0.5: 0.0500,
        1.0: 0.0505,
        2.0: 0.0510,
        3.0: 0.0515,
        5.0: 0.0525,
        7.0: 0.0535,
        10.0: 0.0545,
        20.0: 0.0560,
        30.0: 0.0570,
    }
    curve = bootstrap_zero_curve(sample_par, date(2026, 6, 1))
    console.rule("[bold amber]Curve bootstrap sanity check[/bold amber]")
    for tenor, par_yield in sample_par.items():
        priced = reprice_par_bond(curve, tenor, par_yield)
        status = "OK" if abs(priced - 100.0) < 1e-6 else "FAIL"
        console.log(
            f"tenor {tenor:>5}y  par {par_yield*100:5.2f}%  "
            f"reprices to {priced:.8f}  [{status}]"
        )

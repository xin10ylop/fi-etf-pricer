# Treasury ETF Pricer

A US Treasury ETF fair-value and creation/redemption arbitrage tool. It prices
every bond in an ETF basket itself, from the US Treasury yield curve, rather than
reading prices from a paid vendor. A Python and FastAPI backend does the data
fetching, curve bootstrapping, pricing, and signal calculation; a single-file
dark-terminal dashboard displays the result.

## What it does and why a Fixed Income desk cares

The tool answers a question a creation/redemption desk asks every day: is this
Treasury ETF trading rich or cheap to the value of the bonds it holds, by enough
to cover the cost of doing the arbitrage. To answer it the tool downloads the
published basket for the fund, downloads the same day's US Treasury par yield
curve, and prices each Treasury in the basket from scratch. It bootstraps a zero
curve from the par yields, discounts each bond's known cash flows off that curve,
adds accrued interest, and aggregates the basket into a net asset value per share.
It then compares that NAV to the ETF's closing market price and nets off the real
costs of a creation or redemption to produce a CREATE, REDEEM, or NO TRADE signal.

The defining feature is that it prices the basket itself. Most tools of this kind
read bond prices from a paid market-data vendor. This one does not. It builds the
prices from the free public Treasury curve, which is the part that demonstrates
fixed income pricing skill: bootstrapping spot rates from par yields, handling the
semiannual coupon convention, getting accrued interest and the clean/dirty
ordering right, and proving the curve is self-consistent by repricing par bonds
back to 100.

Because the prices are built from a published, auditable curve, every number the
tool produces can be traced back to its inputs. Each stored mark carries the curve
date and a valuation timestamp, so a reviewer can see exactly which curve produced
which price. That auditability is itself a selling point: the NAV is not a black
box from a vendor feed, it is a calculation anyone can check.

The tool also reports how close its self-computed NAV is to the issuer's official
published NAV. A small tracking number is direct evidence the pricing engine is
working correctly, and it is the most honest way to demonstrate that a
from-scratch pricer matches what the market does in practice.

## The contemporaneous end-of-day design

Bond ETF premiums are mostly an artefact of stale NAV. The bonds an ETF holds are
marked once a day, while the ETF itself trades continuously. Compare a live ETF
price to a NAV struck hours earlier and you will see a premium or discount that is
not real, it is just the two sides being measured at different times.

This tool avoids that trap deliberately. It is an end-of-day tool. The signal
compares the ETF closing price against an NAV built from the same day's closing
Treasury curve, so both sides of the comparison are contemporaneous. Every mark
and price carries a timestamp and the curve date so the contemporaneity is visible
and auditable. Intraday signals are explicitly out of scope for this version.

## How the curve engine works

1. Download the US Treasury Daily Par Yield Curve Rates for the valuation date.
   This is a free public feed with no API key. A FRED alternative is available
   behind a config flag and requires a free FRED API key.
2. Bootstrap a zero-coupon (spot) curve from the par yields. Par yields are the
   coupon rates that make a bond price to 100, not discount rates. The bootstrap
   solves the discount factors sequentially, shortest tenor first, so that a par
   bond at each tenor reprices to exactly 100 under the semiannual convention.
3. Interpolate the zero curve (linear on zero rates) to get a discount factor for
   a cash flow at any time in years, all under semiannual compounding.

Bond pricing then follows a strict order, because the order is a classic source of
error:

1. Build the semiannual cash flow schedule by stepping back from the maturity
   date. Each coupon is the coupon rate over two, per 100 face; the final cash
   flow adds the 100 principal.
2. Discount every future cash flow off the curve. The sum is the dirty price.
   Discounting cash flows always yields the dirty price.
3. Compute accrued interest using the actual/actual convention used by US
   Treasuries: the coupon over two, scaled by actual days from the last coupon to
   settlement over actual days in the coupon period.
4. The clean price is the dirty price minus accrued interest. Never the other way
   around.

A small bid/ask is modelled by widening the clean mid by a configurable
half-spread in price points. This is a modelled assumption documented as such, not
an observed quote.

## NAV, signal, and the tracking check

NAV uses dirty prices, because an ETF NAV includes accrued interest. The basket
dirty value is aggregated in SQL, the cash component from the basket file is added,
and the total is divided by the creation unit size to get NAV per share. The
premium in basis points is the ETF close over NAV minus one. The breakeven band is
the total cost of an arbitrage (creation fee, the bond bid/ask half-spread, and the
ETF trading cost) expressed in basis points of one creation unit. A premium above
the band is a CREATE, a discount below it is a REDEEM, otherwise NO TRADE.

After computing NAV the tool fetches the issuer's official NAV and reports the
tracking difference in basis points. A small number is evidence the engine works.

## Known limitations

- Bid/ask is modelled, not observed. The curve produces a single fair mid and the
  bid/ask is a configurable half-spread assumption, not a real quote.
- T+1 settlement is not modelled. The valuation date is used as the settlement
  date for accrued interest.
- Treasuries only. Corporate, municipal, mortgage, and multi-sector funds need
  credit-spread or vendor pricing this tool does not have, and are rejected.
- End-of-day only. Intraday signals are out of scope, by design (see above).

## Optional FactSet cross-check

A FactSet Fixed Income Prices cross-check is available purely so the dashboard can
show the curve price against a vendor price side by side. It is off by default and
never required. Vendor prices are never substituted into the NAV. To enable it,
install the FactSet SDK packages, set `FACTSET_ENABLED=1`, and point
`FACTSET_CONFIG_PATH` at your FactSet credentials file. Before trusting FactSet
quotes, verify whether they are clean or dirty by pulling one coupon-paying bond
across a coupon date and checking for the sawtooth drop. If the FactSet path is
enabled and fails, the tool logs a warning and continues with the curve price.

## Running the backend

```
cd fi-etf-pricer
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The API serves on http://localhost:8000. Useful endpoints:

- `GET  /api/discover?ticker=IEF` validate a ticker is a Treasury ETF
- `POST /api/watchlist/add` add a ticker (body `{"ticker": "IEF"}`)
- `GET  /api/watchlist` watchlist with latest signals
- `GET  /api/signals/IEF` full detail for one ETF
- `POST /api/run?ticker=IEF` reprice one ETF now
- `POST /api/run-all` reprice the whole watchlist now

You can sanity-check the curve engine and the bond pricer directly:

```
python -m backend.curve     # bootstraps a sample curve, reprices par bonds to 100
python -m backend.pricing   # prices a bond on a flat curve as a hand check
```

The scheduler reprices the whole watchlist daily at 17:30 US Eastern, after the
Treasury curve is published. Run the server in US Eastern time (or adjust the
configured time in `backend/etf_config.py`) so the job fires at the intended hour.

## Deploying

- Frontend: push to `main`. Netlify auto-deploys the dashboard from `frontend/`
  using `netlify.toml`. Set `window.API_BASE` in the page (or edit `index.html`)
  to point at the deployed backend URL.
- Backend: runs on a server such as a DigitalOcean droplet. Install the
  requirements and run `uvicorn backend.main:app` behind a process manager. CORS
  is open so the Netlify frontend can call it.

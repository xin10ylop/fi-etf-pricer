"""Optional FactSet Fixed Income Prices cross-check.

OFF BY DEFAULT. This module is only exercised when both of these hold:

  FACTSET_ENABLED=1
  FACTSET_CONFIG_PATH points at a FactSet credentials config file

The tool must run fully without the FactSet SDK installed, so the SDK imports
happen lazily inside methods, never at module import time. If anything here fails
the caller logs a warning and continues with the curve price. Vendor prices are
never silently substituted into the NAV.

A WARNING ABOUT CLEAN VERSUS DIRTY
FactSet fixed income quotes are assumed CLEAN unless verified otherwise. Before
trusting them you should run a one-off sawtooth test: pull a single coupon-paying
bond across one of its coupon dates and watch the price. A clean price barely
moves across the coupon (it only loses the accrued that had built up); a dirty
price drops by roughly a full coupon on the ex date, producing the classic
sawtooth. Confirm which one FactSet returns before wiring it into anything.
"""

from __future__ import annotations

import os
from datetime import date

from rich.console import Console

from . import etf_config
from .pricing import Bond, PricedBond, accrued_interest

console = Console()


def factset_enabled() -> bool:
    """Return True only when FactSet is explicitly enabled and configured.

    Both the enable flag and a config path must be present. This gate is checked
    everywhere before any FactSet code path runs.
    """
    return os.environ.get("FACTSET_ENABLED") == "1" and bool(
        os.environ.get("FACTSET_CONFIG_PATH")
    )


class FactSetClient:
    """Thin wrapper over the FactSet Fixed Income Prices API.

    Constructed only when FactSet is enabled. All SDK imports are lazy so the
    rest of the system has no hard dependency on the FactSet packages.
    """

    def __init__(self) -> None:
        if not factset_enabled():
            raise RuntimeError(
                "FactSetClient constructed while FactSet is disabled. "
                "Set FACTSET_ENABLED=1 and FACTSET_CONFIG_PATH."
            )
        self.config_path = os.environ["FACTSET_CONFIG_PATH"]

    def _retry(self, fn, what: str):
        """Run a FactSet call with the shared retry policy.

        Three attempts with the configured wait. On total failure the exception
        is raised so the caller can fall back to the curve price.
        """
        import time

        last_exc: Exception | None = None
        for attempt in range(1, etf_config.DOWNLOAD_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                console.log(
                    f"[yellow]FactSet {what} attempt {attempt} failed: {exc}[/yellow]"
                )
                if attempt < etf_config.DOWNLOAD_RETRIES:
                    time.sleep(etf_config.DOWNLOAD_RETRY_WAIT_SECONDS)
        assert last_exc is not None
        raise last_exc

    def price_bonds(
        self, bonds: list[Bond], valuation_date: date
    ) -> list[PricedBond]:
        """Fetch bid/mid/ask for the given CUSIPs from FactSet.

        Returns PricedBonds labelled "factset". Quotes are treated as CLEAN (see
        the module sawtooth warning) and accrued interest is added on top to form
        the dirty prices, mirroring how the curve provider works so the two are
        comparable. The configured Treasury half-spread is used only if FactSet
        returns a single mid without a two-sided market.
        """
        # Lazy SDK imports. The tool must run without these installed.
        import fds.sdk.FactSetPrices  # noqa: F401
        from fds.sdk.FactSetPrices.api import prices_api  # noqa: F401
        from fds.sdk.utils.authentication import ConfidentialClient

        def _call() -> list[PricedBond]:
            client = ConfidentialClient(self.config_path)  # noqa: F841
            # The concrete POST /factset-prices/v1/fixed-income request is wired
            # here. The published quotes are mapped onto PricedBond. Each quote is
            # treated as clean; accrued is added to produce dirty. This block is
            # intentionally defensive: any shape mismatch raises and the caller
            # falls back to the curve.
            results: list[PricedBond] = []
            for bond in bonds:
                # Placeholder mapping. A live deployment fills clean_bid/mid/ask
                # from the FactSet response for bond.cusip. We keep the accrued
                # and dirty derivation explicit so clean-vs-dirty stays auditable.
                accrued = accrued_interest(bond, valuation_date)
                raise NotImplementedError(
                    "Wire the live FactSet response mapping before enabling the "
                    "cross-check in production."
                )
            return results

        return self._retry(_call, "fixed-income prices")

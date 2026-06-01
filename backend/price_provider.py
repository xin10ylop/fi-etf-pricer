"""Pluggable price source abstraction.

The whole point of this module is that the rest of the system (engine, database,
dashboard) never knows where a price came from beyond a short source label. That
lets a paid vendor be slotted in later without touching anything downstream.

PriceProvider is the abstract base. Two implementations ship:

  CurvePriceProvider   default, free. Prices every bond off the bootstrapped
                       Treasury curve. This is what v1 uses.
  FactSetPriceProvider optional, off by default. Pulls bid/mid/ask from FactSet
                       purely for a side-by-side cross-check. Never required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from . import etf_config
from .curve import TreasuryCurve
from .pricing import Bond, PricedBond, price_bond


class PriceProvider(ABC):
    """Abstract base for anything that can price a list of Treasuries.

    Implementations return a PricedBond per input bond, each carrying clean and
    dirty bid/mid/ask, accrued interest, and a source label. Whatever the source,
    the contract is identical so callers stay source agnostic.
    """

    @abstractmethod
    def price_bonds(
        self, bonds: list[Bond], valuation_date: date
    ) -> list[PricedBond]:
        """Return a PricedBond for each bond as of the valuation date."""
        raise NotImplementedError


class CurvePriceProvider(PriceProvider):
    """Default provider. Prices bonds by discounting cash flows off the curve.

    This is the implementation that demonstrates the fixed income pricing skill:
    no vendor prices are read. Each bond is priced with the pricing module against
    a single bootstrapped TreasuryCurve, producing a fair mid plus a modelled
    bid/ask.
    """

    source_label = "curve"

    def __init__(
        self,
        curve: TreasuryCurve,
        half_spread_price: float = etf_config.TREASURY_HALF_SPREAD_PRICE,
    ) -> None:
        self.curve = curve
        self.half_spread_price = half_spread_price

    def price_bonds(
        self, bonds: list[Bond], valuation_date: date
    ) -> list[PricedBond]:
        """Price every bond off the curve as of the valuation date."""
        return [
            price_bond(
                bond,
                self.curve,
                valuation_date,
                half_spread_price=self.half_spread_price,
                source=self.source_label,
            )
            for bond in bonds
        ]


class FactSetPriceProvider(PriceProvider):
    """Optional cross-check provider backed by the FactSet Fixed Income API.

    This is never on the critical path. It is constructed only when FactSet is
    explicitly enabled, and if any call fails the caller is expected to fall back
    to the curve price. It exists so the dashboard can show curve price versus
    FactSet price side by side, never to feed vendor prices into the NAV.
    """

    source_label = "factset"

    def __init__(self) -> None:
        # Imported lazily so the tool runs fully without the FactSet SDK present.
        from .factset_client import FactSetClient

        self.client = FactSetClient()

    def price_bonds(
        self, bonds: list[Bond], valuation_date: date
    ) -> list[PricedBond]:
        """Pull bid/mid/ask for each CUSIP from FactSet for cross-checking."""
        return self.client.price_bonds(bonds, valuation_date)

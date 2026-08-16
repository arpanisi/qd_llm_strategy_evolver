"""Cost models for both tracks (Step 7), composed from config values.

Equities: zipline's VolumeShareSlippage (volume_limit + price_impact) + PerShare
commission, matching the plan's locked parameters (per-share 0.0075, min $1.00,
volume limit 2.5% of ADV, price impact 0.1).

Futures: PerContract commission (2.5/contract) + a custom tick-based slippage
(slippage_ticks of tick_size) that fills at the next close.
"""

from __future__ import annotations

import math

from zipline.finance.commission import PerContract, PerShare
from zipline.finance.slippage import (
    LiquidityExceeded,
    SlippageModel,
    VolumeShareSlippage,
    fill_price_worse_than_limit_price,
    isnull,
)

from src.config.settings import TrackConfig


def equity_commission(track: TrackConfig) -> PerShare:
    cost = track.cost
    return PerShare(cost=cost.per_share_cost, min_trade_cost=cost.min_trade_cost)


def equity_slippage(track: TrackConfig) -> VolumeShareSlippage:
    cost = track.cost
    return VolumeShareSlippage(volume_limit=cost.volume_limit, price_impact=cost.price_impact)


def futures_commission(track: TrackConfig) -> PerContract:
    return PerContract(cost=track.cost.commission_per_contract, exchange_fee=0.0)


def futures_slippage(track: TrackConfig) -> "FuturesSlippage":
    return FuturesSlippage(slippage_ticks=track.cost.slippage_ticks)


class FuturesSlippage(SlippageModel):
    """Fill at next close shifted by ``slippage_ticks`` x tick_size, capped by
    volume limit (1 tick = 0.25 price points on the CME continuous series)."""

    def __init__(self, slippage_ticks: int, volume_limit: float = 1.0):
        super().__init__()
        self.slippage_ticks = int(slippage_ticks)
        self.volume_limit = float(volume_limit)

    def process_order(self, data, order):
        volume = data.current(order.asset, "volume")
        max_volume = self.volume_limit * volume
        remaining_volume = max_volume - self.volume_for_bar
        if remaining_volume < 1:
            raise LiquidityExceeded()
        cur_volume = int(min(remaining_volume, abs(order.open_amount)))
        if cur_volume < 1:
            return None, None

        price = data.current(order.asset, "close")
        if isnull(price):
            return None, None
        tick = getattr(order.asset, "tick_size", 0.25)
        shift = self.slippage_ticks * tick
        impacted_price = price + math.copysign(shift, order.direction)

        if fill_price_worse_than_limit_price(impacted_price, order):
            return None, None

        return impacted_price, math.copysign(cur_volume, order.direction)

"""
Channel Mode Strategy pattern for abstracting General vs AXI mode operations.

Provides unified interfaces for channel-specific operations, reducing
repetitive `if _is_axi_mode:` branches throughout the codebase.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, TYPE_CHECKING

from .flit import AxiChannel

if TYPE_CHECKING:
    from .router import PortWire


class ChannelModeStrategy(ABC):
    """
    Abstract strategy for channel mode operations.

    Provides polymorphic methods for:
    - Getting request/response channel lists
    - Iterating over wires by category
    """

    @property
    @abstractmethod
    def request_channels(self) -> List[AxiChannel]:
        """Get request direction channel list."""
        pass

    @property
    @abstractmethod
    def response_channels(self) -> List[AxiChannel]:
        """Get response direction channel list."""
        pass

    @property
    @abstractmethod
    def all_channels(self) -> List[AxiChannel]:
        """Get all channels in order."""
        pass

    @property
    @abstractmethod
    def channel_count(self) -> int:
        """Number of physical channels (sub-routers)."""
        pass


class GeneralModeStrategy(ChannelModeStrategy):
    """
    General Mode: 2 logical sub-routers (Req + Resp).

    Request channels (AW, W, AR) share the Req sub-router.
    Response channels (B, R) share the Resp sub-router.
    """

    @property
    def request_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR]

    @property
    def response_channels(self) -> List[AxiChannel]:
        return [AxiChannel.B, AxiChannel.R]

    @property
    def all_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR, AxiChannel.B, AxiChannel.R]

    @property
    def channel_count(self) -> int:
        return 2  # Req + Resp


class AXIModeStrategy(ChannelModeStrategy):
    """
    AXI Mode: 5 independent sub-routers (AW, W, AR, B, R).

    Each AXI channel has its own dedicated sub-router,
    eliminating HoL blocking between channels.
    """

    @property
    def request_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR]

    @property
    def response_channels(self) -> List[AxiChannel]:
        return [AxiChannel.B, AxiChannel.R]

    @property
    def all_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR, AxiChannel.B, AxiChannel.R]

    @property
    def channel_count(self) -> int:
        return 5  # AW, W, AR, B, R


# Singleton instances for reuse
GENERAL_MODE_STRATEGY = GeneralModeStrategy()
AXI_MODE_STRATEGY = AXIModeStrategy()


def get_channel_mode_strategy(channel_mode) -> ChannelModeStrategy:
    """
    Get strategy instance for the given channel mode.

    Args:
        channel_mode: ChannelMode enum value.

    Returns:
        Appropriate strategy instance.
    """
    from .router import ChannelMode

    if channel_mode == ChannelMode.AXI:
        return AXI_MODE_STRATEGY
    return GENERAL_MODE_STRATEGY

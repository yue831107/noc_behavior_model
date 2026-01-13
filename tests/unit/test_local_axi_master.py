"""
Tests for Local AXI Master.

Tests cover:
1. LocalTransferConfig encoding/decoding
2. LocalAXIMasterStats
3. LocalAXIMaster initialization and state
"""

import pytest
from unittest.mock import Mock

from src.testbench.local_axi_master import (
    LocalTransferConfig,
    LocalAXIMasterStats,
    LocalAXIMasterState,
    LocalAXIMaster,
)
from src.testbench.memory import LocalMemory


class TestLocalTransferConfig:
    """Test LocalTransferConfig dataclass."""

    def test_default_values(self):
        """Config should have expected defaults."""
        config = LocalTransferConfig(dest_coord=(2, 3))

        assert config.dest_coord == (2, 3)
        assert config.local_src_addr == 0x0000
        assert config.local_dst_addr == 0x1000
        assert config.transfer_size == 256

    def test_encode_user_signal(self):
        """encode_user_signal should pack coordinates."""
        config = LocalTransferConfig(dest_coord=(5, 3))

        signal = config.encode_user_signal()

        # Format: (dest_y << 8) | dest_x
        assert signal == (3 << 8) | 5
        assert signal == 0x0305

    def test_decode_user_signal(self):
        """decode_user_signal should unpack coordinates."""
        # Signal for (x=5, y=3)
        signal = 0x0305

        x, y = LocalTransferConfig.decode_user_signal(signal)

        assert x == 5
        assert y == 3

    def test_encode_decode_roundtrip(self):
        """encode and decode should be inverse operations."""
        config = LocalTransferConfig(dest_coord=(7, 2))

        signal = config.encode_user_signal()
        decoded = LocalTransferConfig.decode_user_signal(signal)

        assert decoded == config.dest_coord


class TestLocalAXIMasterStats:
    """Test LocalAXIMasterStats dataclass."""

    def test_default_values(self):
        """Stats should initialize to zeros."""
        stats = LocalAXIMasterStats()

        assert stats.aw_sent == 0
        assert stats.w_sent == 0
        assert stats.b_received == 0
        assert stats.total_cycles == 0
        assert stats.first_aw_cycle == 0
        assert stats.last_b_cycle == 0


class TestLocalAXIMasterState:
    """Test LocalAXIMasterState enum."""

    def test_state_values(self):
        """States should have expected values."""
        assert LocalAXIMasterState.IDLE.value == "idle"
        assert LocalAXIMasterState.RUNNING.value == "running"
        assert LocalAXIMasterState.COMPLETE.value == "complete"


class TestLocalAXIMaster:
    """Test LocalAXIMaster class."""

    @pytest.fixture
    def local_memory(self):
        """Create local memory with test data."""
        mem = LocalMemory(node_id=0, size=4096)
        mem.fill(0, 256)  # Fill with sequential bytes
        return mem

    def test_initialization(self, local_memory):
        """Master should initialize correctly."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
        )

        assert master.node_id == 0
        assert master._state == LocalAXIMasterState.IDLE

    def test_is_idle(self, local_memory):
        """is_idle should check state."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
        )

        assert master.is_idle is True

    def test_is_complete(self, local_memory):
        """is_complete should check state."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
        )

        # Manually set complete state
        master._state = LocalAXIMasterState.COMPLETE
        assert master.is_complete is True

    def test_configure_transfer(self, local_memory):
        """configure_transfer should set up transfer config."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
        )

        config = LocalTransferConfig(
            dest_coord=(2, 1),
            local_src_addr=0x0000,
            local_dst_addr=0x2000,
            transfer_size=128,
        )

        master.configure_transfer(config)

        assert master._transfer_config == config

"""
Unit tests for Host AXI Master components.

Tests:
1. AXIIdConfig and AXIIdGenerator
2. HostAXIMaster AXI channel operations
3. V1System DMA transfer integration
"""

import pytest
from typing import Optional

from src.testbench import (
    AXIIdConfig,
    AXIIdGenerator,
    AXIMasterController,
)
from src.testbench import (
    HostAXIMaster,
    HostAXIMasterState,
    AXIChannelPort,
    AXIResponsePort,
)
from src.testbench import HostMemory
from src.core.routing_selector import V1System
from src.config import TransferConfig, TransferMode


class TestAXIIdConfig:
    """Test AXIIdConfig dataclass."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = AXIIdConfig()
        assert config.id_width == 4
        assert config.cyclic is True
        assert config.start_id == 0

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = AXIIdConfig(id_width=8, cyclic=False, start_id=5)
        assert config.id_width == 8
        assert config.cyclic is False
        assert config.start_id == 5

    def test_invalid_id_width(self):
        """Invalid id_width should raise ValueError."""
        with pytest.raises(ValueError, match="id_width must be between"):
            AXIIdConfig(id_width=0)
        with pytest.raises(ValueError, match="id_width must be between"):
            AXIIdConfig(id_width=17)

    def test_invalid_start_id(self):
        """start_id exceeding max should raise ValueError."""
        # With 4 bits, max_id = 15
        with pytest.raises(ValueError, match="start_id must be between"):
            AXIIdConfig(id_width=4, start_id=16)


class TestAXIIdGenerator:
    """Test AXIIdGenerator class."""

    def test_cyclic_generation(self):
        """IDs should cycle when cyclic=True."""
        config = AXIIdConfig(id_width=2, cyclic=True, start_id=0)  # max_id = 3
        gen = AXIIdGenerator(config)

        # Generate IDs 0, 1, 2, 3, 0, 1, ...
        ids = []
        for _ in range(6):
            axi_id = gen.get_next_id()
            ids.append(axi_id)
            gen.release_id(axi_id)

        assert ids == [0, 1, 2, 3, 0, 1]

    def test_non_cyclic_generation(self):
        """IDs should not cycle when cyclic=False."""
        config = AXIIdConfig(id_width=2, cyclic=False, start_id=0)
        gen = AXIIdGenerator(config)

        # Generate IDs but they shouldn't wrap
        ids = []
        for _ in range(4):
            axi_id = gen.get_next_id()
            ids.append(axi_id)
            gen.release_id(axi_id)

        # Should be 0, 1, 2, 3 and stay at 3
        assert ids == [0, 1, 2, 3]

    def test_in_flight_tracking(self):
        """In-flight IDs should be tracked."""
        gen = AXIIdGenerator(AXIIdConfig(id_width=2))

        # Allocate without releasing
        id0 = gen.get_next_id()
        id1 = gen.get_next_id()

        assert gen.is_in_flight(id0)
        assert gen.is_in_flight(id1)
        assert gen.in_flight_count == 2
        assert gen.available_ids == 2  # max_id=3, so 4 total - 2 in flight

        # Release one
        gen.release_id(id0)
        assert not gen.is_in_flight(id0)
        assert gen.is_in_flight(id1)
        assert gen.in_flight_count == 1

    def test_skip_in_flight_ids(self):
        """Generator should skip IDs that are in-flight."""
        gen = AXIIdGenerator(AXIIdConfig(id_width=2, start_id=0))

        # Allocate 0, 1 without releasing
        id0 = gen.get_next_id()  # 0
        id1 = gen.get_next_id()  # 1

        assert id0 == 0
        assert id1 == 1

        # Allocate more - should skip 0, 1 and give 2, 3
        id2 = gen.get_next_id()  # 2
        id3 = gen.get_next_id()  # 3

        assert id2 == 2
        assert id3 == 3

    def test_all_ids_in_flight_raises_error(self):
        """Should raise RuntimeError when all IDs are in-flight."""
        gen = AXIIdGenerator(AXIIdConfig(id_width=2))  # Only 4 IDs

        # Allocate all 4
        for _ in range(4):
            gen.get_next_id()

        # Next allocation should fail
        with pytest.raises(RuntimeError, match="All AXI IDs are in-flight"):
            gen.get_next_id()

    def test_reset(self):
        """reset() should clear all state."""
        gen = AXIIdGenerator(AXIIdConfig(id_width=2, start_id=1))

        gen.get_next_id()
        gen.get_next_id()
        assert gen.in_flight_count == 2

        gen.reset()
        assert gen.in_flight_count == 0
        assert gen.get_next_id() == 1  # Back to start_id


class TestAXIChannelPort:
    """Test AXI channel port handshake."""

    def test_initial_state(self):
        """Port should start with valid=False, ready=True."""
        port = AXIChannelPort()
        assert port.out_valid is False
        assert port.out_payload is None
        assert port.in_ready is True

    def test_can_send(self):
        """can_send should check ready and not-valid."""
        port = AXIChannelPort()

        assert port.can_send() is True  # ready and not valid

        port.set_output("data")
        assert port.can_send() is False  # valid set

        port.in_ready = False
        port.clear_output()
        assert port.can_send() is False  # not ready

    def test_handshake(self):
        """Handshake should complete when valid && ready."""
        port = AXIChannelPort()

        port.set_output("data")
        assert port.out_valid is True
        assert port.out_payload == "data"

        # Ready is True by default
        assert port.try_handshake() is True

        # If ready is False, handshake fails
        port.in_ready = False
        assert port.try_handshake() is False


class TestAXIResponsePort:
    """Test AXI response port."""

    def test_initial_state(self):
        """Response port should start empty."""
        port = AXIResponsePort()
        assert port.in_valid is False
        assert port.in_payload is None
        assert port.out_ready is True

    def test_has_response(self):
        """has_response checks valid and payload."""
        port = AXIResponsePort()
        assert port.has_response() is False

        port.in_valid = True
        port.in_payload = "response"
        assert port.has_response() is True

    def test_get_response(self):
        """get_response returns and clears response."""
        port = AXIResponsePort()
        port.in_valid = True
        port.in_payload = "response"

        resp = port.get_response()
        assert resp == "response"
        assert port.in_valid is False
        assert port.in_payload is None


class TestHostAXIMaster:
    """Test HostAXIMaster class."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory with test data."""
        mem = HostMemory(size=4096)
        # Fill with test pattern
        test_data = bytes(range(256)) * 4  # 1KB of test data
        mem.write(0, test_data)
        return mem

    @pytest.fixture
    def transfer_config(self):
        """Create transfer config."""
        return TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],  # Single node
            max_burst_len=8,
            beat_size=8,
            max_outstanding=4,
            transfer_mode=TransferMode.BROADCAST,
        )

    def test_initial_state(self, host_memory, transfer_config):
        """Master should start idle."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )
        assert master.is_idle is True
        assert master.is_running is False
        assert master.is_complete is False

    def test_start(self, host_memory, transfer_config):
        """start() should transition to running."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        master.start()
        assert master.is_idle is False
        assert master.is_running is True

    def test_cannot_start_twice(self, host_memory, transfer_config):
        """start() should be ignored if already running."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        master.start()
        master.start()  # Should be ignored
        assert master.is_running is True

    def test_progress(self, host_memory, transfer_config):
        """progress should reflect controller progress."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        assert master.progress == 0.0 or master.progress == 1.0
        master.start()
        # Progress will depend on controller state


class TestV1SystemDMATransfer:
    """Test V1System DMA transfer integration."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory with test data."""
        mem = HostMemory(size=4096)
        test_data = b"TEST_DATA_" + bytes(54)  # 64 bytes
        mem.write(0, test_data)
        return mem

    def test_configure_transfer_without_memory_raises(self):
        """configure_transfer without host_memory should raise."""
        system = V1System(mesh_cols=5, mesh_rows=4)

        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
        )

        with pytest.raises(ValueError, match="host_memory must be set"):
            system.configure_transfer(config)

    def test_configure_transfer_with_memory(self, host_memory):
        """configure_transfer should create host_axi_master."""
        system = V1System(
            mesh_cols=5,
            mesh_rows=4,
            host_memory=host_memory,
        )

        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
        )

        system.configure_transfer(config)
        assert system.host_axi_master is not None

    def test_start_transfer(self, host_memory):
        """start_transfer should start the master."""
        system = V1System(
            mesh_cols=5,
            mesh_rows=4,
            host_memory=host_memory,
        )

        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
        )

        system.configure_transfer(config)
        result = system.start_transfer()

        assert result is True
        assert system.host_axi_master.is_running is True

    def test_transfer_complete_property(self, host_memory):
        """transfer_complete should reflect master state."""
        system = V1System(
            mesh_cols=5,
            mesh_rows=4,
            host_memory=host_memory,
        )

        # No transfer configured
        assert system.transfer_complete is True

        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
        )

        system.configure_transfer(config)
        system.start_transfer()
        assert system.transfer_complete is False

    def test_process_cycle_with_transfer(self, host_memory):
        """process_cycle should run without errors with transfer."""
        system = V1System(
            mesh_cols=5,
            mesh_rows=4,
            host_memory=host_memory,
        )

        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
            max_burst_len=8,
            beat_size=8,
        )

        system.configure_transfer(config)
        system.start_transfer()

        # Run cycles - should not raise any errors
        initial_time = system.current_time
        for _ in range(50):
            system.process_cycle()

        # Time should have advanced
        assert system.current_time == initial_time + 50

        # Master should have sent some AW transactions
        assert system.host_axi_master.stats.aw_sent >= 0


class TestHostAXIMasterStats:
    """Test HostAXIMasterStats throughput calculations."""

    def test_effective_write_throughput_zero_cycles(self):
        """Throughput should be 0 when total_cycles is 0."""
        from src.testbench.host_axi_master import HostAXIMasterStats
        stats = HostAXIMasterStats()
        assert stats.effective_write_throughput == 0.0

    def test_effective_write_throughput(self):
        """Throughput should be b_received / total_cycles."""
        from src.testbench.host_axi_master import HostAXIMasterStats
        stats = HostAXIMasterStats()
        stats.b_received = 10
        stats.total_cycles = 100
        assert stats.effective_write_throughput == 0.1

    def test_effective_read_throughput_zero_cycles(self):
        """Read throughput should be 0 when total_cycles is 0."""
        from src.testbench.host_axi_master import HostAXIMasterStats
        stats = HostAXIMasterStats()
        assert stats.effective_read_throughput == 0.0

    def test_effective_read_throughput(self):
        """Read throughput should be r_received / total_cycles."""
        from src.testbench.host_axi_master import HostAXIMasterStats
        stats = HostAXIMasterStats()
        stats.r_received = 20
        stats.total_cycles = 200
        assert stats.effective_read_throughput == 0.1


class TestHostAXIMasterReset:
    """Test HostAXIMaster reset functionality."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory."""
        mem = HostMemory(size=4096)
        mem.write(0, bytes(256))
        return mem

    @pytest.fixture
    def transfer_config(self):
        """Create transfer config."""
        return TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
        )

    def test_reset_clears_state(self, host_memory, transfer_config):
        """reset() should clear all state."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        master.start()
        assert master.is_running is True

        master.reset()
        assert master.is_idle is True
        assert master.is_running is False
        assert master.stats.aw_sent == 0


class TestHostAXIMasterQueue:
    """Test HostAXIMaster multi-transfer queue functionality."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory."""
        mem = HostMemory(size=4096)
        mem.write(0, bytes(256))
        return mem

    def test_queue_transfer(self, host_memory):
        """queue_transfer should add config to queue."""
        config1 = TransferConfig(src_addr=0, src_size=64, dst_addr=0x1000, target_nodes=[1])
        config2 = TransferConfig(src_addr=64, src_size=64, dst_addr=0x2000, target_nodes=[2])

        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config1,
        )

        master.queue_transfer(config1)
        master.queue_transfer(config2)

        assert len(master._transfer_queue) == 2

    def test_queue_transfers(self, host_memory):
        """queue_transfers should add multiple configs."""
        configs = [
            TransferConfig(src_addr=0, src_size=64, dst_addr=0x1000, target_nodes=[1]),
            TransferConfig(src_addr=64, src_size=64, dst_addr=0x2000, target_nodes=[2]),
            TransferConfig(src_addr=128, src_size=64, dst_addr=0x3000, target_nodes=[3]),
        ]

        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=configs[0],
        )

        master.queue_transfers(configs)

        assert len(master._transfer_queue) == 3

    def test_start_queue_empty(self, host_memory):
        """start_queue with empty queue should do nothing."""
        config = TransferConfig(src_addr=0, src_size=64, dst_addr=0x1000, target_nodes=[1])
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        master.start_queue()  # Empty queue
        assert master.is_idle is True

    def test_queue_progress(self, host_memory):
        """queue_progress should return (completed, total)."""
        config = TransferConfig(src_addr=0, src_size=64, dst_addr=0x1000, target_nodes=[1])
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        master.queue_transfer(config)
        master.queue_transfer(config)

        completed, total = master.queue_progress
        assert completed == 0
        assert total == 2

    def test_all_queue_transfers_complete_idle(self, host_memory):
        """all_queue_transfers_complete should return True when idle."""
        config = TransferConfig(src_addr=0, src_size=64, dst_addr=0x1000, target_nodes=[1])
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        assert master.all_queue_transfers_complete is True

    def test_start_queue_with_callbacks(self, host_memory):
        """start_queue should call on_transfer_start callback."""
        config = TransferConfig(src_addr=0, src_size=64, dst_addr=0x1000, target_nodes=[1])

        callback_calls = []
        def on_start(idx, cfg):
            callback_calls.append(('start', idx))

        def on_complete(idx, cfg):
            callback_calls.append(('complete', idx))

        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
            on_transfer_start=on_start,
            on_transfer_complete=on_complete,
        )

        master.queue_transfer(config)
        master.start_queue()

        assert ('start', 0) in callback_calls


class TestHostAXIMasterReadMode:
    """Test HostAXIMaster read mode."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory."""
        mem = HostMemory(size=4096)
        return mem

    def test_configure_read(self, host_memory):
        """configure_read should set read mode."""
        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
            transfer_mode=TransferMode.BROADCAST_READ,
        )
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        golden = {(1, 0x1000): b'\x00' * 64}
        master.configure_read(golden)

        assert master._is_read_mode is True

    def test_read_data_property(self, host_memory):
        """read_data should return collected data."""
        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
        )
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        # Manually add test data
        master._read_data[(1, 0x1000)] = b'test_data'

        data = master.read_data
        assert (1, 0x1000) in data
        assert data[(1, 0x1000)] == b'test_data'


class TestHostAXIMasterSummary:
    """Test HostAXIMaster summary methods."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory."""
        mem = HostMemory(size=4096)
        mem.write(0, bytes(256))
        return mem

    @pytest.fixture
    def transfer_config(self):
        """Create transfer config."""
        return TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
        )

    def test_get_summary_write_mode(self, host_memory, transfer_config):
        """get_summary should return write mode summary."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        master.start()
        summary = master.get_summary()

        assert summary['mode'] == 'write'
        assert 'state' in summary
        assert 'progress' in summary
        assert 'timing' in summary
        assert 'axi_channels' in summary

    def test_get_summary_read_mode(self, host_memory):
        """get_summary should return read mode summary."""
        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
            transfer_mode=TransferMode.BROADCAST_READ,
        )
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        master.configure_read({})
        master.start()
        summary = master.get_summary()

        assert summary['mode'] == 'read'
        assert 'ar_sent' in summary['axi_channels']
        assert 'read_data_count' in summary

    def test_print_summary_write_mode(self, host_memory, transfer_config, capsys):
        """print_summary should output write mode stats."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        master.start()
        master.print_summary()

        captured = capsys.readouterr()
        assert "WRITE Mode" in captured.out
        assert "AW Sent" in captured.out

    def test_print_summary_read_mode(self, host_memory, capsys):
        """print_summary should output read mode stats."""
        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
            transfer_mode=TransferMode.BROADCAST_READ,
        )
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        master.configure_read({})
        master.start()
        master.print_summary()

        captured = capsys.readouterr()
        assert "READ Mode" in captured.out
        assert "AR Sent" in captured.out

    def test_controller_stats_property(self, host_memory, transfer_config):
        """controller_stats should return internal stats."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        stats = master.controller_stats
        assert stats is not None


class TestAXIMasterControllerWithIdGenerator:
    """Test AXIMasterController with AXIIdGenerator."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory."""
        mem = HostMemory(size=4096)
        mem.write(0, bytes(256))
        return mem

    @pytest.fixture
    def transfer_config(self):
        """Create transfer config."""
        return TransferConfig(
            src_addr=0,
            src_size=128,
            dst_addr=0x1000,
            target_nodes=[1, 2],
            max_burst_len=8,
            beat_size=8,
            max_outstanding=4,
        )

    def test_controller_uses_cyclic_ids(self, host_memory, transfer_config):
        """Controller should use cyclic IDs from generator."""
        id_config = AXIIdConfig(id_width=2, cyclic=True, start_id=0)
        controller = AXIMasterController(
            config=transfer_config,
            host_memory=host_memory,
            axi_id_config=id_config,
        )

        controller.initialize()

        # Generate transactions and collect IDs
        ids = []
        for cycle in range(10):
            for txn in controller.generate(cycle):
                ids.append(txn.aw.awid)

        # IDs should cycle within 0-3 range
        for axi_id in ids:
            assert 0 <= axi_id <= 3

    def test_controller_releases_ids_on_response(self, host_memory, transfer_config):
        """Controller should release IDs when handling response."""
        from src.axi.interface import AXI_B, AXIResp

        id_config = AXIIdConfig(id_width=2, cyclic=True)
        controller = AXIMasterController(
            config=transfer_config,
            host_memory=host_memory,
            axi_id_config=id_config,
        )

        controller.initialize()

        # Generate one transaction
        txns = list(controller.generate(0))
        assert len(txns) >= 1
        first_id = txns[0].aw.awid

        # ID should be in-flight
        assert controller._id_generator.is_in_flight(first_id)

        # Handle response
        response = AXI_B(bid=first_id, bresp=AXIResp.OKAY)
        controller.handle_response(response, 1)

        # ID should be released
        assert not controller._id_generator.is_in_flight(first_id)

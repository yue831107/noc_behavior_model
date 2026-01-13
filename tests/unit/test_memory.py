"""
Tests for Memory models.

Tests cover:
1. Memory class - basic read/write operations
2. Memory class - fill methods
3. Memory class - file I/O methods
4. HostMemory initialization
5. LocalMemory initialization
6. MemoryCopyDescriptor properties
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.testbench.memory import (
    Memory,
    MemoryConfig,
    MemoryStats,
    HostMemory,
    LocalMemory,
    MemoryCopyDescriptor,
)


class TestMemoryInitialization:
    """Test Memory initialization."""

    def test_default_initialization(self):
        """Memory should initialize with default config."""
        mem = Memory()
        assert mem.name == "Memory"
        assert mem.config.size == 0x100000000
        assert mem.config.latency_read == 1
        assert mem.config.latency_write == 1
        assert mem.used_bytes == 0

    def test_custom_initialization(self):
        """Memory should accept custom config."""
        config = MemoryConfig(
            size=1024,
            latency_read=5,
            latency_write=3,
            bus_width=16,
        )
        mem = Memory(config, name="TestMem")
        assert mem.name == "TestMem"
        assert mem.config.size == 1024
        assert mem.config.latency_read == 5
        assert mem.config.latency_write == 3


class TestMemoryReadWrite:
    """Test Memory read/write operations."""

    def test_write_and_read(self):
        """Should write and read data correctly."""
        mem = Memory(MemoryConfig(size=1024))

        data = bytes([0x12, 0x34, 0x56, 0x78])
        latency = mem.write(0x100, data)

        assert latency == 1
        assert mem.stats.writes == 1
        assert mem.stats.bytes_written == 4

        result, read_latency = mem.read(0x100, 4)
        assert result == data
        assert read_latency == 1
        assert mem.stats.reads == 1
        assert mem.stats.bytes_read == 4

    def test_read_unwritten_address_returns_zeros(self):
        """Reading unwritten addresses should return zeros."""
        mem = Memory(MemoryConfig(size=1024))

        result, _ = mem.read(0, 8)
        assert result == bytes(8)

    def test_write_address_out_of_range_raises(self):
        """Write beyond memory size should raise ValueError."""
        mem = Memory(MemoryConfig(size=256))

        with pytest.raises(ValueError, match="Address out of range"):
            mem.write(250, bytes(10))

    def test_write_negative_address_raises(self):
        """Write to negative address should raise ValueError."""
        mem = Memory(MemoryConfig(size=256))

        with pytest.raises(ValueError, match="Address out of range"):
            mem.write(-1, bytes(4))

    def test_read_address_out_of_range_raises(self):
        """Read beyond memory size should raise ValueError."""
        mem = Memory(MemoryConfig(size=256))

        with pytest.raises(ValueError, match="Address out of range"):
            mem.read(250, 10)

    def test_read_negative_address_raises(self):
        """Read from negative address should raise ValueError."""
        mem = Memory(MemoryConfig(size=256))

        with pytest.raises(ValueError, match="Address out of range"):
            mem.read(-1, 4)


class TestMemoryFill:
    """Test Memory fill methods."""

    def test_fill_sequential_pattern(self):
        """fill() without pattern should use sequential bytes."""
        mem = Memory(MemoryConfig(size=1024))

        mem.fill(0, 16)

        data = mem.get_contents(0, 16)
        expected = bytes(range(16))
        assert data == expected

    def test_fill_sequential_wraps_at_256(self):
        """Sequential pattern should wrap at 0xFF."""
        mem = Memory(MemoryConfig(size=2048))

        mem.fill(0, 300)

        data = mem.get_contents(256, 4)
        assert data == bytes([0, 1, 2, 3])  # Wrapped

    def test_fill_custom_pattern(self):
        """fill() with pattern should repeat the pattern."""
        mem = Memory(MemoryConfig(size=1024))

        pattern = bytes([0xAA, 0x55])
        mem.fill(0, 8, pattern)

        data = mem.get_contents(0, 8)
        assert data == bytes([0xAA, 0x55] * 4)

    def test_fill_random(self):
        """fill_random() should produce reproducible random data."""
        mem1 = Memory(MemoryConfig(size=1024))
        mem2 = Memory(MemoryConfig(size=1024))

        mem1.fill_random(0, 32, seed=42)
        mem2.fill_random(0, 32, seed=42)

        data1 = mem1.get_contents(0, 32)
        data2 = mem2.get_contents(0, 32)

        assert data1 == data2
        assert len(set(data1)) > 1  # Not all same bytes


class TestMemoryClear:
    """Test Memory clear operation."""

    def test_clear_removes_all_data(self):
        """clear() should remove all written data."""
        mem = Memory(MemoryConfig(size=1024))

        mem.write(0, bytes([1, 2, 3, 4]))
        mem.write(100, bytes([5, 6, 7, 8]))
        assert mem.used_bytes == 8

        mem.clear()

        assert mem.used_bytes == 0
        data, _ = mem.read(0, 4)
        assert data == bytes(4)

    def test_clear_resets_stats(self):
        """clear() should reset statistics."""
        mem = Memory(MemoryConfig(size=1024))

        mem.write(0, bytes(4))
        mem.read(0, 4)

        mem.clear()

        assert mem.stats.reads == 0
        assert mem.stats.writes == 0
        assert mem.stats.bytes_read == 0
        assert mem.stats.bytes_written == 0


class TestMemoryVerify:
    """Test Memory verify operation."""

    def test_verify_matching_data_returns_true(self):
        """verify() should return True for matching data."""
        mem = Memory(MemoryConfig(size=1024))

        data = bytes([0x11, 0x22, 0x33, 0x44])
        mem.write(0x100, data)

        assert mem.verify(0x100, data) is True

    def test_verify_mismatched_data_returns_false(self):
        """verify() should return False for mismatched data."""
        mem = Memory(MemoryConfig(size=1024))

        mem.write(0x100, bytes([0x11, 0x22, 0x33, 0x44]))

        assert mem.verify(0x100, bytes([0x11, 0x22, 0x33, 0x55])) is False


class TestMemoryGetContents:
    """Test Memory get_contents method."""

    def test_get_contents_does_not_update_stats(self):
        """get_contents() should not affect statistics."""
        mem = Memory(MemoryConfig(size=1024))

        mem.write(0, bytes([1, 2, 3, 4]))
        initial_reads = mem.stats.reads

        _ = mem.get_contents(0, 4)

        assert mem.stats.reads == initial_reads


class TestMemoryUsedBytes:
    """Test Memory used_bytes property."""

    def test_used_bytes_reflects_written_data(self):
        """used_bytes should count unique written addresses."""
        mem = Memory(MemoryConfig(size=1024))

        assert mem.used_bytes == 0

        mem.write(0, bytes(10))
        assert mem.used_bytes == 10

        mem.write(100, bytes(5))
        assert mem.used_bytes == 15


class TestMemoryFileIO:
    """Test Memory file I/O methods."""

    def test_load_from_file(self):
        """load_from_file() should load binary data."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            test_data = bytes([0x12, 0x34, 0x56, 0x78, 0x9A])
            path.write_bytes(test_data)

            mem = Memory(MemoryConfig(size=1024))
            loaded = mem.load_from_file(path, address=0x100)

            assert loaded == 5
            data = mem.get_contents(0x100, 5)
            assert data == test_data

    def test_load_from_file_not_found_raises(self):
        """load_from_file() should raise FileNotFoundError."""
        mem = Memory(MemoryConfig(size=1024))

        with pytest.raises(FileNotFoundError, match="File not found"):
            mem.load_from_file("nonexistent.bin")

    def test_load_from_file_exceeds_size_raises(self):
        """load_from_file() should raise ValueError if data too large."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.bin"
            path.write_bytes(bytes(200))

            mem = Memory(MemoryConfig(size=100))

            with pytest.raises(ValueError, match="exceeds available memory"):
                mem.load_from_file(path)

    def test_dump_to_file(self):
        """dump_to_file() should write binary data."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dump.bin"

            mem = Memory(MemoryConfig(size=1024))
            test_data = bytes([0xAA, 0xBB, 0xCC, 0xDD])
            mem.write(0, test_data)

            written = mem.dump_to_file(path, address=0, size=4)

            assert written == 4
            assert path.read_bytes() == test_data

    def test_dump_to_file_clamps_size(self):
        """dump_to_file() should clamp size to valid range."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dump.bin"

            mem = Memory(MemoryConfig(size=100))
            mem.fill(0, 100)

            # Request more than available
            written = mem.dump_to_file(path, address=90, size=100)

            assert written == 10  # Clamped to remaining bytes

    def test_dump_to_hex(self):
        """dump_to_hex() should create formatted hex file."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dump.hex"

            mem = Memory(MemoryConfig(size=1024))
            mem.write(0, bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F]))  # "Hello"

            mem.dump_to_hex(path, address=0, size=5, bytes_per_line=16)

            content = path.read_text()
            assert "00000000" in content
            assert "48 65 6C 6C 6F" in content
            assert "Hello" in content

    def test_dump_used_regions_empty_memory(self):
        """dump_used_regions() should return 0 for empty memory."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sparse.bin"

            mem = Memory(MemoryConfig(size=1024))

            written = mem.dump_used_regions(path)

            assert written == 0

    def test_dump_used_regions(self):
        """dump_used_regions() should dump only written regions."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sparse.bin"

            mem = Memory(MemoryConfig(size=10000))
            mem.write(1000, bytes([1, 2, 3]))
            mem.write(1010, bytes([4, 5]))

            written = mem.dump_used_regions(path)

            # Should dump from addr 1000 to 1011 (inclusive)
            assert written == 12
            data = path.read_bytes()
            assert data[0] == 1
            assert data[10] == 4


class TestHostMemory:
    """Test HostMemory class."""

    def test_default_initialization(self):
        """HostMemory should use appropriate defaults."""
        mem = HostMemory()

        assert mem.name == "HostMemory"
        assert mem.config.size == 1024 * 1024  # 1MB
        assert mem.config.latency_read == 10
        assert mem.config.latency_write == 10

    def test_custom_initialization(self):
        """HostMemory should accept custom parameters."""
        mem = HostMemory(size=4096, latency_read=5, latency_write=3)

        assert mem.config.size == 4096
        assert mem.config.latency_read == 5
        assert mem.config.latency_write == 3

    def test_inherits_memory_operations(self):
        """HostMemory should support Memory operations."""
        mem = HostMemory(size=1024)

        mem.write(0, bytes([1, 2, 3, 4]))
        data, _ = mem.read(0, 4)

        assert data == bytes([1, 2, 3, 4])


class TestLocalMemory:
    """Test LocalMemory class."""

    def test_default_initialization(self):
        """LocalMemory should use appropriate defaults."""
        mem = LocalMemory(node_id=5)

        assert mem.name == "LocalMemory_5"
        assert mem.node_id == 5
        assert mem.config.size == 0x100000000  # 4GB
        assert mem.config.latency_read == 1
        assert mem.config.latency_write == 1

    def test_custom_initialization(self):
        """LocalMemory should accept custom parameters."""
        mem = LocalMemory(node_id=3, size=2048, latency_read=2, latency_write=2)

        assert mem.node_id == 3
        assert mem.config.size == 2048
        assert mem.config.latency_read == 2

    def test_inherits_memory_operations(self):
        """LocalMemory should support Memory operations."""
        mem = LocalMemory(node_id=0, size=1024)

        mem.fill(0, 16)
        data = mem.get_contents(0, 16)

        assert data == bytes(range(16))


class TestMemoryCopyDescriptor:
    """Test MemoryCopyDescriptor dataclass."""

    def test_initialization(self):
        """MemoryCopyDescriptor should initialize with required fields."""
        desc = MemoryCopyDescriptor(
            src_addr=0x1000,
            dst_node=3,
            dst_addr=0x2000,
            size=256,
        )

        assert desc.src_addr == 0x1000
        assert desc.dst_node == 3
        assert desc.dst_addr == 0x2000
        assert desc.size == 256
        assert desc.block_size == 64  # Default
        assert desc.bytes_sent == 0
        assert desc.bytes_acked == 0

    def test_is_complete_false_initially(self):
        """is_complete should be False when bytes_acked < size."""
        desc = MemoryCopyDescriptor(
            src_addr=0, dst_node=0, dst_addr=0, size=100
        )

        assert desc.is_complete is False

        desc.bytes_acked = 50
        assert desc.is_complete is False

    def test_is_complete_true_when_done(self):
        """is_complete should be True when bytes_acked >= size."""
        desc = MemoryCopyDescriptor(
            src_addr=0, dst_node=0, dst_addr=0, size=100
        )

        desc.bytes_acked = 100
        assert desc.is_complete is True

        desc.bytes_acked = 150  # Over-acked
        assert desc.is_complete is True

    def test_progress_calculation(self):
        """progress should return correct percentage."""
        desc = MemoryCopyDescriptor(
            src_addr=0, dst_node=0, dst_addr=0, size=200
        )

        assert desc.progress == 0.0

        desc.bytes_acked = 100
        assert desc.progress == 50.0

        desc.bytes_acked = 200
        assert desc.progress == 100.0

    def test_progress_zero_size(self):
        """progress should return 100% for zero size."""
        desc = MemoryCopyDescriptor(
            src_addr=0, dst_node=0, dst_addr=0, size=0
        )

        assert desc.progress == 100.0

    def test_latency_calculation(self):
        """latency should return end_cycle - start_cycle."""
        desc = MemoryCopyDescriptor(
            src_addr=0, dst_node=0, dst_addr=0, size=100
        )

        desc.start_cycle = 100
        assert desc.latency == 0  # end_cycle not set

        desc.end_cycle = 250
        assert desc.latency == 150


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_default_initialization(self):
        """MemoryStats should initialize with zeros."""
        stats = MemoryStats()

        assert stats.reads == 0
        assert stats.writes == 0
        assert stats.bytes_read == 0
        assert stats.bytes_written == 0

    def test_custom_initialization(self):
        """MemoryStats should accept custom values."""
        stats = MemoryStats(reads=10, writes=5, bytes_read=100, bytes_written=50)

        assert stats.reads == 10
        assert stats.writes == 5
        assert stats.bytes_read == 100
        assert stats.bytes_written == 50

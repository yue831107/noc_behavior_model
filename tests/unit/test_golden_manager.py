"""
Unit tests for GoldenManager.
"""

import pytest
from pathlib import Path
import tempfile

from src.core.golden_manager import (
    GoldenManager,
    GoldenEntry,
    GoldenSource,
    VerificationResult,
    VerificationReport,
)


class TestGoldenEntry:
    """Tests for GoldenEntry dataclass."""

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        entry = GoldenEntry(
            node_id=5,
            local_addr=0x1000,
            data=b"test_data_123",
            source=GoldenSource.WRITE_CAPTURE,
            capture_cycle=100,
        )

        data = entry.to_dict()
        restored = GoldenEntry.from_dict(data)

        assert restored.node_id == entry.node_id
        assert restored.local_addr == entry.local_addr
        assert restored.data == entry.data
        assert restored.source == entry.source
        assert restored.capture_cycle == entry.capture_cycle


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_passed_result(self):
        """Test passed verification result."""
        result = VerificationResult(
            node_id=0,
            local_addr=0x1000,
            expected=b"test",
            actual=b"test",
            passed=True,
        )

        assert result.passed
        assert result.size_match
        assert result.first_mismatch_offset == -1

    def test_failed_result_data_mismatch(self):
        """Test failed verification with data mismatch."""
        result = VerificationResult(
            node_id=0,
            local_addr=0x1000,
            expected=b"test",
            actual=b"tXst",
            passed=False,
            first_mismatch_offset=1,
        )

        assert not result.passed
        assert result.size_match
        assert result.first_mismatch_offset == 1

    def test_failed_result_size_mismatch(self):
        """Test failed verification with size mismatch."""
        result = VerificationResult(
            node_id=0,
            local_addr=0x1000,
            expected=b"test",
            actual=b"te",
            passed=False,
            first_mismatch_offset=2,
        )

        assert not result.passed
        assert not result.size_match
        assert result.expected_size == 4
        assert result.actual_size == 2


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_empty_report(self):
        """Test empty report properties."""
        report = VerificationReport()

        assert report.total_checks == 0
        assert report.pass_rate == 0.0
        assert report.all_passed

    def test_all_passed_report(self):
        """Test all passed report."""
        report = VerificationReport(
            total_checks=10,
            passed=10,
            failed=0,
        )

        assert report.pass_rate == 1.0
        assert report.all_passed

    def test_some_failed_report(self):
        """Test partially failed report."""
        report = VerificationReport(
            total_checks=10,
            passed=7,
            failed=3,
        )

        assert report.pass_rate == 0.7
        assert not report.all_passed


class TestGoldenManager:
    """Tests for GoldenManager class."""

    def test_capture_write(self):
        """Test capturing golden data during write."""
        manager = GoldenManager()

        manager.capture_write(
            node_id=0,
            addr=0x1000,
            data=b"test_data",
            cycle=100,
        )

        assert manager.entry_count == 1
        golden = manager.get_golden(0, 0x1000)
        assert golden == b"test_data"

    def test_set_golden_manual(self):
        """Test manually setting golden data."""
        manager = GoldenManager()

        manager.set_golden(
            node_id=5,
            addr=0x2000,
            data=b"manual_golden",
            source=GoldenSource.MANUAL,
        )

        assert manager.entry_count == 1
        golden = manager.get_golden(5, 0x2000)
        assert golden == b"manual_golden"

    def test_get_golden_not_found(self):
        """Test getting non-existent golden data."""
        manager = GoldenManager()
        assert manager.get_golden(0, 0x1000) is None

    def test_get_golden_store(self):
        """Test getting all golden data as dict."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"data1", 0)
        manager.capture_write(1, 0x2000, b"data2", 0)

        store = manager.get_golden_store()

        assert len(store) == 2
        assert store[(0, 0x1000)] == b"data1"
        assert store[(1, 0x2000)] == b"data2"

    def test_clear(self):
        """Test clearing all golden data."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"data", 0)
        manager.clear()

        assert manager.entry_count == 0

    def test_verify_all_pass(self):
        """Test verification with all matches."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"data1", 0)
        manager.capture_write(1, 0x2000, b"data2", 0)

        read_results = {
            (0, 0x1000): b"data1",
            (1, 0x2000): b"data2",
        }

        report = manager.verify(read_results)

        assert report.total_checks == 2
        assert report.passed == 2
        assert report.failed == 0
        assert report.all_passed

    def test_verify_data_mismatch(self):
        """Test verification with data mismatch."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"expected", 0)

        read_results = {
            (0, 0x1000): b"actual__",  # Different data
        }

        report = manager.verify(read_results)

        assert report.total_checks == 1
        assert report.passed == 0
        assert report.failed == 1
        assert not report.all_passed
        assert len(report.results) == 1
        assert report.results[0].first_mismatch_offset == 0

    def test_verify_missing_actual(self):
        """Test verification with missing actual data."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"expected", 0)

        read_results = {}  # No read data

        report = manager.verify(read_results)

        assert report.total_checks == 1
        assert report.missing_actual == 1
        assert report.failed == 1

    def test_verify_missing_golden(self):
        """Test verification with missing golden data."""
        manager = GoldenManager()

        read_results = {
            (0, 0x1000): b"unexpected",  # No golden for this
        }

        report = manager.verify(read_results)

        assert report.total_checks == 1
        assert report.missing_golden == 1
        assert report.failed == 1

    def test_generate_report_text_all_passed(self):
        """Test report generation when all passed."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"data", 0)

        report = manager.verify({(0, 0x1000): b"data"})
        text = manager.generate_report_text(report)

        assert "ALL CHECKS PASSED" in text
        assert "Pass Rate: 100.0%" in text

    def test_generate_report_text_with_failures(self):
        """Test report generation with failures."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"expected", 0)

        report = manager.verify({(0, 0x1000): b"actual__"})
        text = manager.generate_report_text(report)

        assert "FAILURES:" in text
        assert "Node  0" in text
        assert "First mismatch" in text

    def test_save_and_load_file(self):
        """Test saving and loading golden data to/from file."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"data1", 100)
        manager.capture_write(1, 0x2000, b"data2", 200)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "golden.json"
            manager.save_to_file(path)

            # Create new manager and load
            manager2 = GoldenManager()
            count = manager2.load_from_file(path)

            assert count == 2
            assert manager2.get_golden(0, 0x1000) == b"data1"
            assert manager2.get_golden(1, 0x2000) == b"data2"

    def test_get_summary(self):
        """Test summary generation."""
        manager = GoldenManager()
        manager.capture_write(0, 0x1000, b"data1", 0)
        manager.set_golden(1, 0x2000, b"data2", GoldenSource.MANUAL)

        summary = manager.get_summary()

        assert summary["entry_count"] == 2
        assert "write_capture" in summary["sources"]
        assert "manual" in summary["sources"]

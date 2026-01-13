"""
Golden Data Manager for Read-back Verification.

Provides auto-capture of golden data during write operations and
detailed verification/comparison reports for read-back tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import json

# Type alias for golden key: (node_id, addr) or ("host", addr)
GoldenKey = Tuple[Union[int, str], int]


class GoldenSource(Enum):
    """Source of golden data."""
    WRITE_CAPTURE = "write_capture"  # Auto-captured during write
    FILE = "file"                     # Loaded from file
    MANUAL = "manual"                 # Manually set


@dataclass
class GoldenEntry:
    """
    Single golden data entry.

    Supports two key types:
    - (node_id: int, addr: int) for LocalMemory targets
    - ("host", addr: int) for HostMemory targets (used in GATHER)
    """
    node_id: Union[int, str]  # int for node, "host" for HostMemory
    local_addr: int
    data: bytes
    source: GoldenSource
    capture_cycle: int = 0

    @property
    def is_host_memory(self) -> bool:
        """Check if this entry targets HostMemory."""
        return self.node_id == "host"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "local_addr": self.local_addr,
            "data": self.data.hex(),
            "source": self.source.value,
            "capture_cycle": self.capture_cycle,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GoldenEntry":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            local_addr=data["local_addr"],
            data=bytes.fromhex(data["data"]),
            source=GoldenSource(data["source"]),
            capture_cycle=data.get("capture_cycle", 0),
        )


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    node_id: Union[int, str]  # int for node, "host" for HostMemory
    local_addr: int
    expected: bytes
    actual: bytes
    passed: bool
    first_mismatch_offset: int = -1  # -1 if passed

    @property
    def expected_size(self) -> int:
        """Get expected data size."""
        return len(self.expected)

    @property
    def actual_size(self) -> int:
        """Get actual data size."""
        return len(self.actual)

    @property
    def size_match(self) -> bool:
        """Check if sizes match."""
        return len(self.expected) == len(self.actual)


@dataclass
class VerificationReport:
    """Complete verification report."""
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    missing_golden: int = 0
    missing_actual: int = 0
    results: List[VerificationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_checks == 0:
            return 0.0
        return self.passed / self.total_checks

    @property
    def all_passed(self) -> bool:
        """Check if all verifications passed."""
        return self.failed == 0 and self.missing_golden == 0 and self.missing_actual == 0


class GoldenManager:
    """
    Golden Data Manager for Read-back Verification.

    Captures golden data during write operations and provides detailed
    verification reports for read-back tests.

    Usage:
        # During write operation
        manager = GoldenManager()
        manager.capture_write(node_id=0, addr=0x1000, data=b'...', cycle=100)

        # After read operation
        report = manager.verify(read_results)
        manager.print_report(report)
    """

    def __init__(self):
        """Initialize GoldenManager."""
        # Golden store: (node_id, local_addr) -> GoldenEntry
        # Key can be (int, int) for LocalMemory or ("host", int) for HostMemory
        self._golden_store: Dict[GoldenKey, GoldenEntry] = {}

    def capture_write(
        self,
        node_id: Union[int, str],
        addr: int,
        data: bytes,
        cycle: int = 0
    ) -> None:
        """
        Capture golden data during write operation.

        Args:
            node_id: Target node ID (int) or "host" for HostMemory.
            addr: Memory address.
            data: Data being written.
            cycle: Simulation cycle when captured.
        """
        key: GoldenKey = (node_id, addr)
        entry = GoldenEntry(
            node_id=node_id,
            local_addr=addr,
            data=data,
            source=GoldenSource.WRITE_CAPTURE,
            capture_cycle=cycle,
        )
        self._golden_store[key] = entry

    def capture_gather(
        self,
        host_addr: int,
        data_portions: List[bytes],
        cycle: int = 0
    ) -> None:
        """
        Capture golden data for GATHER operation (concatenated to HostMemory).

        Args:
            host_addr: HostMemory destination address.
            data_portions: List of data portions from each node (in order).
            cycle: Simulation cycle when captured.
        """
        concatenated = b"".join(data_portions)
        key: GoldenKey = ("host", host_addr)
        entry = GoldenEntry(
            node_id="host",
            local_addr=host_addr,
            data=concatenated,
            source=GoldenSource.WRITE_CAPTURE,
            capture_cycle=cycle,
        )
        self._golden_store[key] = entry

    def set_golden(
        self,
        node_id: Union[int, str],
        addr: int,
        data: bytes,
        source: GoldenSource = GoldenSource.MANUAL
    ) -> None:
        """
        Manually set golden data.

        Args:
            node_id: Target node ID (int) or "host" for HostMemory.
            addr: Memory address.
            data: Expected data.
            source: Source of the golden data.
        """
        key: GoldenKey = (node_id, addr)
        entry = GoldenEntry(
            node_id=node_id,
            local_addr=addr,
            data=data,
            source=source,
        )
        self._golden_store[key] = entry

    def get_golden(
        self,
        node_id: Union[int, str],
        addr: int
    ) -> Optional[bytes]:
        """
        Get golden data for a specific location.

        Args:
            node_id: Target node ID (int) or "host" for HostMemory.
            addr: Memory address.

        Returns:
            Golden data bytes or None if not found.
        """
        key: GoldenKey = (node_id, addr)
        entry = self._golden_store.get(key)
        return entry.data if entry else None

    def get_host_golden(self, addr: int) -> Optional[bytes]:
        """
        Get golden data for HostMemory location.

        Args:
            addr: HostMemory address.

        Returns:
            Golden data bytes or None if not found.
        """
        return self.get_golden("host", addr)

    def get_golden_store(self) -> Dict[GoldenKey, bytes]:
        """
        Get all golden data as simple dictionary.

        Returns:
            Dict of (node_id, local_addr) -> data bytes.
        """
        return {key: entry.data for key, entry in self._golden_store.items()}

    def clear(self) -> None:
        """Clear all golden data."""
        self._golden_store.clear()

    def verify(
        self,
        read_results: Dict[GoldenKey, bytes]
    ) -> VerificationReport:
        """
        Verify read results against golden data.

        Args:
            read_results: Dict of (node_id, local_addr) -> actual data.

        Returns:
            VerificationReport with detailed results.
        """
        report = VerificationReport()

        # Check all golden entries
        all_keys = set(self._golden_store.keys()) | set(read_results.keys())
        report.total_checks = len(all_keys)

        for key in sorted(all_keys):
            node_id, addr = key
            golden_entry = self._golden_store.get(key)
            actual_data = read_results.get(key)

            if golden_entry is None:
                # Missing golden data
                report.missing_golden += 1
                report.failed += 1
                if actual_data is not None:
                    result = VerificationResult(
                        node_id=node_id,
                        local_addr=addr,
                        expected=b"",
                        actual=actual_data,
                        passed=False,
                        first_mismatch_offset=0,
                    )
                    report.results.append(result)
                continue

            if actual_data is None:
                # Missing actual data
                report.missing_actual += 1
                report.failed += 1
                result = VerificationResult(
                    node_id=node_id,
                    local_addr=addr,
                    expected=golden_entry.data,
                    actual=b"",
                    passed=False,
                    first_mismatch_offset=0,
                )
                report.results.append(result)
                continue

            # Compare data
            expected = golden_entry.data
            passed = expected == actual_data
            first_mismatch = -1

            if not passed:
                # Find first mismatch
                for i in range(min(len(expected), len(actual_data))):
                    if expected[i] != actual_data[i]:
                        first_mismatch = i
                        break
                if first_mismatch == -1 and len(expected) != len(actual_data):
                    # Size mismatch
                    first_mismatch = min(len(expected), len(actual_data))

            result = VerificationResult(
                node_id=node_id,
                local_addr=addr,
                expected=expected,
                actual=actual_data,
                passed=passed,
                first_mismatch_offset=first_mismatch,
            )
            report.results.append(result)

            if passed:
                report.passed += 1
            else:
                report.failed += 1

        return report

    def generate_report_text(
        self,
        report: VerificationReport,
        show_data_bytes: int = 64
    ) -> str:
        """
        Generate human-readable report text.

        Args:
            report: Verification report.
            show_data_bytes: Max bytes to show in data preview.

        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("READ-BACK VERIFICATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Total Checks: {report.total_checks}")
        lines.append(f"Passed: {report.passed}")
        lines.append(f"Failed: {report.failed}")
        if report.missing_golden > 0:
            lines.append(f"Missing Golden: {report.missing_golden}")
        if report.missing_actual > 0:
            lines.append(f"Missing Actual: {report.missing_actual}")
        lines.append(f"Pass Rate: {report.pass_rate * 100:.1f}%")
        lines.append("")

        if report.all_passed:
            lines.append("ALL CHECKS PASSED!")
        else:
            lines.append("FAILURES:")
            lines.append("-" * 60)

            for result in report.results:
                if result.passed:
                    continue

                if result.node_id == "host":
                    lines.append(
                        f"  HostMemory @ 0x{result.local_addr:08X}:"
                    )
                else:
                    lines.append(
                        f"  Node {result.node_id:2d} @ 0x{result.local_addr:08X}:"
                    )

                if not result.expected:
                    lines.append("    [MISSING GOLDEN DATA]")
                    lines.append(
                        f"    Actual ({result.actual_size} bytes): "
                        f"{self._format_hex(result.actual, show_data_bytes)}"
                    )
                elif not result.actual:
                    lines.append("    [MISSING ACTUAL DATA]")
                    lines.append(
                        f"    Expected ({result.expected_size} bytes): "
                        f"{self._format_hex(result.expected, show_data_bytes)}"
                    )
                else:
                    lines.append(
                        f"    First mismatch at offset: {result.first_mismatch_offset}"
                    )
                    if not result.size_match:
                        lines.append(
                            f"    Size mismatch: expected {result.expected_size}, "
                            f"got {result.actual_size}"
                        )
                    lines.append(
                        f"    Expected ({result.expected_size} bytes): "
                        f"{self._format_hex(result.expected, show_data_bytes)}"
                    )
                    lines.append(
                        f"    Actual ({result.actual_size} bytes): "
                        f"{self._format_hex(result.actual, show_data_bytes)}"
                    )

                lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def print_report(
        self,
        report: VerificationReport,
        show_data_bytes: int = 64
    ) -> None:
        """Print verification report."""
        print(self.generate_report_text(report, show_data_bytes))

    def _format_hex(self, data: bytes, max_bytes: int) -> str:
        """Format bytes as hex string with truncation."""
        if len(data) <= max_bytes:
            return data.hex()
        return data[:max_bytes].hex() + "..."

    def save_to_file(self, path: Path) -> None:
        """
        Save golden data to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "golden_entries": [
                entry.to_dict()
                for entry in self._golden_store.values()
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: Path) -> int:
        """
        Load golden data from JSON file.

        Args:
            path: Input file path.

        Returns:
            Number of entries loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for entry_data in data.get("golden_entries", []):
            entry = GoldenEntry.from_dict(entry_data)
            key = (entry.node_id, entry.local_addr)
            self._golden_store[key] = entry
            count += 1

        return count

    @property
    def entry_count(self) -> int:
        """Get number of golden entries."""
        return len(self._golden_store)

    @property
    def entries(self) -> List[GoldenEntry]:
        """Get all golden entries."""
        return list(self._golden_store.values())

    def get_summary(self) -> Dict:
        """Get golden manager summary."""
        sources = {}
        for entry in self._golden_store.values():
            src = entry.source.value
            sources[src] = sources.get(src, 0) + 1

        return {
            "entry_count": self.entry_count,
            "sources": sources,
        }

    def generate_noc_golden(
        self,
        node_configs: List,
        get_node_memory,
        mesh_cols: int = 5,
    ) -> int:
        """
        Generate golden data for NoC-to-NoC traffic.

        For each transfer config:
          1. Read source data from Node[src].LocalMemory[src_addr]
          2. Store as golden for Node[dst].LocalMemory[dst_addr]

        Collision handling: Last write wins based on route distance.
        When multiple sources write to the same destination, the source
        with the longest route (last to complete) determines the final value.

        Args:
            node_configs: List of NodeTransferConfig objects.
            get_node_memory: Callable(node_id) -> Memory to get node's memory.
            mesh_cols: Mesh columns (for coord_to_node_id conversion).

        Returns:
            Number of golden entries generated.
        """
        count = 0
        compute_cols = mesh_cols - 1
        
        def get_src_coord(src_node_id: int):
            """Convert node ID to coordinate."""
            x = (src_node_id % compute_cols) + 1
            y = src_node_id // compute_cols
            return (x, y)
        
        def get_route_distance(src_node_id: int, dest_coord):
            """Calculate Manhattan distance for XY routing."""
            src_coord = get_src_coord(src_node_id)
            return abs(src_coord[0] - dest_coord[0]) + abs(src_coord[1] - dest_coord[1])
        
        # Sort by route distance, then by src_node_id (ascending for ties)
        # Higher src_node_id wins in simulation for most TIE cases,
        # so we process them LAST (they overwrite lower src_node_id)
        sorted_configs = sorted(
            node_configs,
            key=lambda nc: (get_route_distance(nc.src_node_id, nc.dest_coord), nc.src_node_id)
        )

        for nc in sorted_configs:
            # Calculate destination node_id from coord
            dest_x, dest_y = nc.dest_coord
            if dest_x < 1 or dest_x >= mesh_cols:
                continue  # Edge column, skip

            dst_node_id = (dest_x - 1) + dest_y * compute_cols

            # Read expected data from source node's memory
            src_memory = get_node_memory(nc.src_node_id)
            expected_data, _ = src_memory.read(
                nc.local_src_addr, nc.transfer_size
            )

            # Store as golden for destination node
            # Later entries (longer routes) will overwrite earlier ones
            self.capture_write(
                node_id=dst_node_id,
                addr=nc.local_dst_addr,
                data=expected_data,
                cycle=0,
            )
            count += 1

        return count


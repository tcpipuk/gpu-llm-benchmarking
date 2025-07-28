"""NVIDIA GPU monitoring for benchmarking.

This module provides the GPUMonitor class for comprehensive GPU metrics collection
during benchmark execution, including utilization, memory, temperature, and power data.
"""

from __future__ import annotations

import csv
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from threading import Event, Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

# GPU monitoring constants
GPU_QUERY_FIELDS = (
    "utilization.gpu,utilization.memory,memory.used,memory.free,"
    "temperature.gpu,power.draw,clocks.current.graphics,"
    "clocks.current.memory,fan.speed,pstate"
)
GPU_QUERY_FIELD_COUNT = 10
PSTATE_INDEX = 9
GPU_METRIC_COUNT = 8
GPU_CSV_HEADERS = [
    "timestamp",
    "gpu_util",
    "gpu_mem_util",
    "vram_used_mb",
    "vram_free_mb",
    "temperature_c",
    "power_watts",
    "clock_graphics_mhz",
    "clock_memory_mhz",
    "fan_speed_percent",
    "pstate",
]


class GPUMonitor:
    """Monitors comprehensive GPU metrics during benchmark execution.

    Captures GPU utilization, memory bandwidth, VRAM usage, temperature,
    power consumption, clock speeds, and performance states at 0.5s intervals.
    """

    def __init__(self, output_file: Path) -> None:
        """Initialise GPU monitor with output file for metrics storage.

        Args:
            output_file: Path to store GPU monitoring metrics
        """
        self.output_file = output_file
        self.metrics: list[tuple] = []
        self.stop_event = Event()
        self.thread: Thread | None = None

    def _get_gpu_stats(self) -> tuple:
        """Get comprehensive GPU metrics via nvidia-smi.

        Returns:
            Tuple of (gpu_util, mem_util, vram_used, vram_free, temp, power,
                     clock_graphics, clock_memory, fan_speed, pstate).
        """
        try:
            result = subprocess.run(
                [
                    "/usr/bin/nvidia-smi",
                    f"--query-gpu={GPU_QUERY_FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return self._default_gpu_stats()

            values = [x.strip() for x in result.stdout.strip().split(",")]
            if len(values) < GPU_QUERY_FIELD_COUNT:
                return self._default_gpu_stats()

            return self._parse_gpu_values(values)

        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return self._default_gpu_stats()

    def _default_gpu_stats(self) -> tuple:
        """Return default GPU stats when nvidia-smi is unavailable."""
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "P0")

    def _parse_gpu_values(self, values: list[str]) -> tuple:
        """Parse nvidia-smi output values into appropriate types.

        Returns:
            Tuple of parsed values.
        """
        parsed_values = []
        for i, val in enumerate(values):
            if val in {"N/A", "[N/A]"}:
                parsed_values.append(0.0 if i < GPU_METRIC_COUNT else "P0")  # Default for pstate
            elif i == PSTATE_INDEX:  # pstate is string
                parsed_values.append(val)
            else:
                try:
                    parsed_values.append(float(val))
                except ValueError:
                    parsed_values.append(0.0)
        return tuple(parsed_values)

    def _monitor_loop(self) -> None:
        """Continuous monitoring loop that runs in background thread.

        Captures GPU metrics every 0.5 seconds and writes to CSV file.
        """
        with Path(self.output_file).open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(GPU_CSV_HEADERS)

            while not self.stop_event.is_set():
                timestamp = time.time()
                stats = self._get_gpu_stats()
                self.metrics.append((timestamp, *stats))
                writer.writerow([timestamp, *list(stats)])
                f.flush()
                time.sleep(0.5)

    @contextmanager
    def monitor(self) -> Generator[None, None, None]:
        """Context manager for GPU monitoring during test execution.

        Yields:
            None while monitoring is active.
        """
        self.metrics.clear()
        self.stop_event.clear()
        self.thread = Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        try:
            yield
        finally:
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=2)

    def get_stats(self) -> tuple:
        """Calculate comprehensive statistics from collected metrics.

        Returns:
            Tuple of min/avg/max values for all monitored metrics and most common pstate.
        """
        if not self.metrics:
            return (0.0,) * 16 + ("P0",)

        # Extract each metric type (skip timestamp)
        gpu_utils = [m[1] for m in self.metrics]
        gpu_mem_utils = [m[2] for m in self.metrics]
        vram_used = [m[3] for m in self.metrics]
        vram_free = [m[4] for m in self.metrics]
        temps = [m[5] for m in self.metrics]
        powers = [m[6] for m in self.metrics]
        clock_graphics = [m[7] for m in self.metrics]
        clock_memory = [m[8] for m in self.metrics]
        fan_speeds = [m[9] for m in self.metrics]
        pstates = [m[10] for m in self.metrics]

        # Find most common pstate
        most_common_pstate = max(set(pstates), key=pstates.count) if pstates else "P0"

        return (
            mean(gpu_utils),
            max(gpu_utils),
            min(gpu_utils),
            mean(gpu_mem_utils),
            max(gpu_mem_utils),
            min(gpu_mem_utils),
            mean(vram_used),
            max(vram_used),
            min(vram_used),
            mean(vram_free),
            max(vram_free),
            min(vram_free),
            mean(temps),
            max(temps),
            min(temps),
            mean(powers),
            max(powers),
            min(powers),
            mean(clock_graphics),
            mean(clock_memory),
            mean(fan_speeds),
            most_common_pstate,
        )

"""Results management for GPU benchmarking.

This module provides the ResultsManager class for handling benchmark results
storage, file operations, system information capture, and summary generation.
"""

from __future__ import annotations

import csv
import logging
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING

from .logger import logger
from .models import HF_MODEL

if TYPE_CHECKING:
    from .models import TestResult

# Constants for summary generation
MIN_RUNS_FOR_STDEV = 2

# Ollama configuration constants
OLLAMA_CONFIG = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KEEP_ALIVE": "10m",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_NUM_GPU": "1",
    "OLLAMA_NUM_PARALLEL": "1",
}


class ResultsManager:
    """Manages benchmark results storage, file operations, and summary generation.

    Handles CSV export, system information capture, logging setup, and
    statistical analysis of benchmark results.
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialise results manager with base directory for results storage."""
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[TestResult] = []

        # Use the shared logger - no need to reconfigure it
        self.logger = logger

        # Add file handler for benchmark log only
        file_handler = logging.FileHandler(self.base_dir / "benchmark.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)

    def save_system_info(self) -> None:
        """Capture and save comprehensive system information.

        Collects GPU specs, CPU info, memory details, software versions,
        and Ollama configuration for reproducibility and analysis context.
        """
        info_file = self.base_dir / "system_info.txt"

        try:
            with Path(info_file).open("w", encoding="utf-8") as f:
                f.write("=== System Information ===\n")
                f.write(f"Date: {datetime.now(tz=UTC)}\n\n")

                # Ollama configuration
                f.write("=== Ollama Configuration ===\n")
                f.writelines(f"{key}: {value}\n" for key, value in OLLAMA_CONFIG.items())
                f.write(f"Model: {HF_MODEL}\n")

                # Ollama version
                try:
                    ollama_version = subprocess.run(
                        ["ollama", "--version"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    f.write(f"Ollama Version: {ollama_version.stdout.strip()}\n\n")
                except subprocess.TimeoutExpired:
                    f.write("Ollama Version: timeout\n\n")
                except Exception as e:
                    f.write(f"Ollama Version: error - {e}\n\n")

                # GPU info
                try:
                    gpu_info = subprocess.run(
                        ["nvidia-smi", "-q"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    f.write("=== GPU Information ===\n")
                    f.write(gpu_info.stdout)
                except subprocess.TimeoutExpired:
                    f.write("GPU information timeout\n")

                # System info
                for cmd, label in [
                    (["lscpu"], "CPU Information"),
                    (["free", "-h"], "Memory Information"),
                ]:
                    try:
                        result = subprocess.run(
                            cmd, check=False, capture_output=True, text=True, timeout=5
                        )
                        f.write(f"\n=== {label} ===\n")
                        f.write(result.stdout)
                    except subprocess.TimeoutExpired:
                        f.write(f"\n{label} timeout\n")

        except Exception:
            self.logger.exception("Failed to save system info")

    def add_result(self, result: TestResult) -> None:
        """Add a test result to the collection."""
        self.results.append(result)

    def save_results(self) -> None:
        """Save all results to CSV files.

        Exports individual test results and generates summary statistics.
        """
        # Individual results
        individual_file = self.base_dir / "individual_results.csv"
        with Path(individual_file).open("w", encoding="utf-8", newline="") as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))

        # Summary statistics
        self._save_summary_stats()

    def _save_summary_stats(self) -> None:
        """Generate and save summary statistics grouped by scenario and context length.

        Calculates means and standard deviations for all metrics across test runs.
        """
        summary_file = self.base_dir / "summary_stats.csv"

        # Group results by scenario and context length
        groups: dict[tuple[str, int], list[TestResult]] = {}
        for result in self.results:
            key = (result.scenario, result.context_length)
            groups.setdefault(key, []).append(result)

        with Path(summary_file).open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scenario",
                "context_length",
                "avg_tokens_per_second",
                "std_tokens_per_second",
                "avg_gpu_util",
                "std_gpu_util",
                "avg_gpu_mem_util",
                "std_gpu_mem_util",
                "avg_vram_used_gb",
                "std_vram_used_gb",
                "avg_vram_free_gb",
                "std_vram_free_gb",
                "avg_eval_time_ms",
                "std_eval_time_ms",
                "avg_response_length",
                "avg_temp_c",
                "avg_power_w",
                "avg_graphics_clock_mhz",
                "avg_memory_clock_mhz",
                "run_count",
            ])

            for (scenario, context_length), results in sorted(groups.items()):
                if len(results) < MIN_RUNS_FOR_STDEV:
                    continue

                tokens_ps = [r.tokens_per_second for r in results]
                gpu_utils = [r.gpu_util_avg for r in results]
                gpu_mem_utils = [r.gpu_mem_util_avg for r in results]
                vrams_used = [r.vram_used_avg_mb / 1024 for r in results]  # Convert to GB
                vrams_free = [r.vram_free_avg_mb / 1024 for r in results]  # Convert to GB
                eval_times = [r.eval_duration_ms for r in results]
                response_lens = [r.response_length_chars for r in results]
                temps = [r.gpu_temp_avg for r in results]
                powers = [r.gpu_power_avg for r in results]
                graphics_clocks = [r.gpu_clock_graphics_avg for r in results]
                memory_clocks = [r.gpu_clock_memory_avg for r in results]

                writer.writerow([
                    scenario,
                    context_length,
                    round(mean(tokens_ps), 3),
                    round(stdev(tokens_ps), 3),
                    round(mean(gpu_utils), 1),
                    round(stdev(gpu_utils), 1),
                    round(mean(gpu_mem_utils), 1),
                    round(stdev(gpu_mem_utils), 1),
                    round(mean(vrams_used), 2),
                    round(stdev(vrams_used), 2),
                    round(mean(vrams_free), 2),
                    round(stdev(vrams_free), 2),
                    round(mean(eval_times), 1),
                    round(stdev(eval_times), 1),
                    round(mean(response_lens)),
                    round(mean(temps), 1),
                    round(mean(powers), 1),
                    round(mean(graphics_clocks)),
                    round(mean(memory_clocks)),
                    len(results),
                ])

    def print_summary(self) -> None:
        """Print comprehensive benchmark summary to console.

        Displays results grouped by scenario with statistical analysis
        and validates benchmark completion status.
        """
        if not self.results:
            logger.warning("No results to summarise")
            return

        # Group by scenario
        scenarios: dict[str, list[TestResult]] = {}
        for result in self.results:
            scenarios.setdefault(result.scenario, []).append(result)

        logger.info("\n=== BENCHMARK COMPLETE ===")

        logger.info("=== FULL BENCHMARK RESULTS ===")

        for scenario, results in scenarios.items():
            logger.info("=== %s SCENARIO RESULTS ===", scenario.upper())
            logger.info(
                "Context Length | Tokens/sec (±SD) | GPU Util% (±SD) | Mem Util% (±SD) | "
                "VRAM Used GB (±SD)"
            )

            # Group by context length
            contexts: dict[int, list[TestResult]] = {}
            for result in results:
                contexts.setdefault(result.context_length, []).append(result)

            for context_length in sorted(contexts.keys()):
                ctx_results = contexts[context_length]

                tokens_ps = [r.tokens_per_second for r in ctx_results]
                gpu_utils = [r.gpu_util_avg for r in ctx_results]
                gpu_mem_utils = [r.gpu_mem_util_avg for r in ctx_results]
                vrams_used = [r.vram_used_avg_mb / 1024 for r in ctx_results]

                if len(ctx_results) > 1:
                    tokens_mean, tokens_std = mean(tokens_ps), stdev(tokens_ps)
                    gpu_mean, gpu_std = mean(gpu_utils), stdev(gpu_utils)
                    gpu_mem_mean, gpu_mem_std = (
                        mean(gpu_mem_utils),
                        stdev(gpu_mem_utils),
                    )
                    vram_mean, vram_std = mean(vrams_used), stdev(vrams_used)
                else:
                    tokens_mean, tokens_std = tokens_ps[0], 0
                    gpu_mean, gpu_std = gpu_utils[0], 0
                    gpu_mem_mean, gpu_mem_std = gpu_mem_utils[0], 0
                    vram_mean, vram_std = vrams_used[0], 0

                logger.info(
                    "%11s | %8.1f±%.1f | %8.1f±%.1f | %8.1f±%.1f | %7.2f±%.2f",
                    context_length,
                    tokens_mean,
                    tokens_std,
                    gpu_mean,
                    gpu_std,
                    gpu_mem_mean,
                    gpu_mem_std,
                    vram_mean,
                    vram_std,
                )

        logger.info("✓ Full benchmark suite complete!")
        logger.info("✓ Ready for community analysis and sharing")

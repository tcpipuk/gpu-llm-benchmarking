#!/usr/bin/env python3
"""NVIDIA GPU LLM Context Length Benchmark.

Comprehensive benchmarking tool for testing Large Language Model inference performance
across different context window sizes on NVIDIA GPUs. Tests two scenarios: short prompts
(allocation overhead) and half-context prompts (realistic usage) to measure GPU utilization,
memory bandwidth, VRAM usage, and generation speed. Designed to evaluate GPU performance
across different NVIDIA architectures with detailed monitoring of all GPU metrics and
comprehensive data collection for analysis.

Tests across different context windows with two scenarios:
1. Short prompts (allocation overhead testing)
2. Half-context prompts (realistic usage testing)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from helpers.logger import logger
from helpers.models import (
    CONTEXT_END,
    CONTEXT_MULTIPLIER,
    CONTEXT_START,
    HF_MODEL,
    TEST_ITERATIONS,
    LocalBenchmarkConfig,
    TestResult,
)
from helpers.nvidia_gpu_monitor import GPUMonitor
from helpers.ollama_client import OllamaClient
from helpers.prompt_generator import PromptGenerator
from helpers.results_manager import ResultsManager

OUTPUT_PATH = os.getenv("OUTPUT_PATH", ".")  # Directory for benchmark results

# Ollama configuration - critical for reproducible results
OLLAMA_CONFIG = {
    "OLLAMA_FLASH_ATTENTION": os.getenv("OLLAMA_FLASH_ATTENTION", "1"),
    "OLLAMA_KEEP_ALIVE": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
    "OLLAMA_KV_CACHE_TYPE": os.getenv("OLLAMA_KV_CACHE_TYPE", "q8_0"),
    "OLLAMA_MAX_LOADED_MODELS": os.getenv("OLLAMA_MAX_LOADED_MODELS", "1"),
    "OLLAMA_NUM_GPU": os.getenv("OLLAMA_NUM_GPU", "1"),
    "OLLAMA_NUM_PARALLEL": os.getenv("OLLAMA_NUM_PARALLEL", "1"),
}

# Logger imported from helpers.logger


# Remaining constants
MIN_RUNS_FOR_STDEV = 2


class BenchmarkRunner:
    """Main benchmark execution coordinator.

    Orchestrates the complete benchmark process including environment setup,
    model warmup, test execution, and results collection across all scenarios
    and context lengths.
    """

    def __init__(self, config: LocalBenchmarkConfig) -> None:
        """Initialise benchmark runner with configuration and results manager."""
        self.config = config
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(OUTPUT_PATH)
        self.results_dir = output_dir / f"benchmark_results_{timestamp}"
        self.results_manager = ResultsManager(self.results_dir)
        self.ollama_client = OllamaClient()

        logger.info("🎯 Benchmark initialised")
        logger.info("📁 Results directory: %s", self.results_dir)
        logger.info("🚀 Running FULL benchmark suite")

    def setup_environment(self) -> None:
        """Setup environment for benchmark execution.

        Assumes Ollama is already installed and running with the model loaded.
        Logs version for reproducibility and verifies connection.

        Raises:
            RuntimeError: If Ollama is not available or model not loaded.
        """
        logger.info("⚙️ Setting up benchmark environment...")

        # Log Ollama version for reproducibility
        try:
            version_result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, check=True
            )
            logger.info("📝 Ollama version: %s", version_result.stdout.strip())
        except subprocess.CalledProcessError as e:
            logger.warning("⚠️ Could not get Ollama version: %s", e)

        # Verify Ollama is running by testing connection
        logger.info("🔗 Verifying Ollama connection...")
        try:
            warmup_response = self.ollama_client.generate(
                model=self.config.model,
                prompt="Hello",
                context_length=self.config.context_start,
            )
            warmup_tokens = warmup_response.get("eval_count", 0)
            logger.info("✅ Ollama connection verified - generated %s tokens", warmup_tokens)
        except Exception as e:
            logger.exception("❌ Ollama connection failed")
            msg = "Ollama is not available or model not loaded"
            raise RuntimeError(msg) from e

        # Save system info
        self.results_manager.save_system_info()

    def run_single_test(self, scenario: str, context_length: int, run_number: int) -> TestResult:
        """Run a single benchmark test with comprehensive monitoring.

        Args:
            scenario: Test scenario ('short' or 'half_context')
            context_length: Context window size for this test
            run_number: Current run iteration number

        Returns:
            TestResult object containing all metrics and timing data.
        """
        test_id = f"{scenario}_{context_length}_{run_number}_{int(time.time())}"

        logger.info(
            "🧪 Running test: %s scenario, %s context, run %s/%s",
            scenario,
            context_length,
            run_number,
            self.config.runs_per_context,
        )

        # Generate prompt and save it
        prompt = PromptGenerator.generate(scenario, context_length, self.config)
        prompt_file = self.results_dir / f"prompt_{test_id}.txt"
        prompt_file.write_text(prompt)

        # Setup GPU monitoring
        monitor_file = self.results_dir / f"gpu_monitor_{test_id}.csv"
        gpu_monitor = GPUMonitor(monitor_file)

        # Run test with monitoring
        start_time = time.time()

        with gpu_monitor.monitor():
            ollama_response = self.ollama_client.generate(
                model=self.config.model, prompt=prompt, context_length=context_length
            )

        end_time = time.time()

        # Save full response
        response_file = self.results_dir / f"ollama_response_{test_id}.json"
        response_file.write_text(json.dumps(ollama_response, indent=2))

        # Save generated text
        if "response" in ollama_response:
            text_file = self.results_dir / f"response_{test_id}.txt"
            text_file.write_text(ollama_response["response"])

        # Calculate metrics
        eval_count = ollama_response.get("eval_count", 0)
        eval_duration = ollama_response.get("eval_duration", 0)

        tokens_per_second = (eval_count * 1_000_000_000 / eval_duration) if eval_duration > 0 else 0

        gpu_stats = gpu_monitor.get_stats()

        # Create result with expanded GPU metrics
        result = TestResult(
            timestamp=datetime.now(tz=UTC).isoformat(),
            scenario=scenario,
            context_length=context_length,
            run_number=run_number,
            tokens_per_second=tokens_per_second,
            total_time_seconds=end_time - start_time,
            prompt_eval_count=ollama_response.get("prompt_eval_count", 0),
            prompt_eval_duration_ms=ollama_response.get("prompt_eval_duration", 0) / 1_000_000,
            eval_count=eval_count,
            eval_duration_ms=eval_duration / 1_000_000,
            total_duration_ms=ollama_response.get("total_duration", 0) / 1_000_000,
            load_duration_ms=ollama_response.get("load_duration", 0) / 1_000_000,
            response_length_chars=len(ollama_response.get("response", "")),
            gpu_util_avg=gpu_stats[0],
            gpu_util_max=gpu_stats[1],
            gpu_util_min=gpu_stats[2],
            gpu_mem_util_avg=gpu_stats[3],
            gpu_mem_util_max=gpu_stats[4],
            gpu_mem_util_min=gpu_stats[5],
            vram_used_avg_mb=gpu_stats[6],
            vram_used_max_mb=gpu_stats[7],
            vram_used_min_mb=gpu_stats[8],
            vram_free_avg_mb=gpu_stats[9],
            vram_free_max_mb=gpu_stats[10],
            vram_free_min_mb=gpu_stats[11],
            gpu_temp_avg=gpu_stats[12],
            gpu_temp_max=gpu_stats[13],
            gpu_temp_min=gpu_stats[14],
            gpu_power_avg=gpu_stats[15],
            gpu_power_max=gpu_stats[16],
            gpu_power_min=gpu_stats[17],
            gpu_clock_graphics_avg=gpu_stats[18],
            gpu_clock_memory_avg=gpu_stats[19],
            gpu_fan_speed_avg=gpu_stats[20],
            gpu_pstate_mode=gpu_stats[21],
            test_id=test_id,
        )

        logger.info(
            "  ✅ Generated %s tokens in %.1fms (%.1f t/s) | GPU: %.1f%% | "
            "MemBW: %.1f%% | VRAM: %.1fGB",
            eval_count,
            eval_duration / 1_000_000,
            tokens_per_second,
            gpu_stats[0],
            gpu_stats[3],
            gpu_stats[6] / 1024,
        )

        return result

    def run_benchmark(self) -> None:
        """Run complete benchmark suite across all scenarios and context lengths.

        Executes all configured test combinations, collects results,
        and generates comprehensive summary output.
        """
        total_tests = (
            len(self.config.scenarios or [])
            * len(self.config.context_lengths or [])
            * self.config.runs_per_context
        )

        logger.info("🎬 Starting benchmark with %s total tests", total_tests)
        test_count = 0

        for scenario in self.config.scenarios or []:
            logger.info("🎭 === %s SCENARIO ===", scenario.upper())

            for context_length in self.config.context_lengths or []:
                logger.info("📏 --- Context Length: %s ---", context_length)

                for run_number in range(1, self.config.runs_per_context + 1):
                    test_count += 1

                    try:
                        result = self.run_single_test(scenario, context_length, run_number)
                        self.results_manager.add_result(result)

                        logger.info("✅ Test %s/%s complete", test_count, total_tests)

                        # Brief pause between tests
                        time.sleep(3)

                    except Exception:
                        logger.exception("❌ Test failed")
                        continue

        # Save results and print summary
        self.results_manager.save_results()
        self.results_manager.print_summary()

        logger.info("🎉 Full benchmark complete! Results saved to: %s", self.results_dir)
        logger.info("📊 Complete dataset ready for community sharing")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for benchmark configuration.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LLM Benchmark with Context Length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Configuration (edit globals at top of script):
  Model: {HF_MODEL}
  Iterations per test: {TEST_ITERATIONS}
  Context range: {CONTEXT_START} → {CONTEXT_END} (x{CONTEXT_MULTIPLIER} each step)
  Output path: {OUTPUT_PATH}
        """,
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for benchmark execution.

    Initialises configuration and runs the complete benchmark suite with error handling and logging.
    """
    parse_arguments()  # Parse arguments for help message support

    config = LocalBenchmarkConfig()
    runner = BenchmarkRunner(config)

    # Log configuration
    total_tests = (
        len(config.scenarios or []) * len(config.context_lengths or []) * config.runs_per_context
    )
    logger.info("🚀 Starting full benchmark suite")
    logger.info("🤖 Model: %s", config.model)
    logger.info("📏 Context lengths: %s", config.context_lengths)
    logger.info("🔄 Iterations per context: %s", config.runs_per_context)
    logger.info("📁 Output path: %s", OUTPUT_PATH)
    logger.info("🧮 Total tests: %s", total_tests)

    # Log Ollama configuration
    logger.info("⚙️ Ollama configuration:")
    for key, value in OLLAMA_CONFIG.items():
        logger.info("  %s: %s", key, value)

    logger.info(
        "📊 Testing %s → %s (x%s progression)",
        config.context_start,
        config.context_end,
        config.context_multiplier,
    )

    try:
        runner.setup_environment()
        runner.run_benchmark()
    except KeyboardInterrupt:
        logger.info("⏹️ Benchmark interrupted by user")
    except Exception:
        logger.exception("💥 Benchmark failed")
        raise


if __name__ == "__main__":
    main()

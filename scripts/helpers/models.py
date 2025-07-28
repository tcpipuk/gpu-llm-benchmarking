"""Data models and configuration classes for GPU LLM benchmarking.

This module contains all dataclasses and configuration models used across
the benchmarking system, providing a single source of truth for data structures.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass
class RemoteBenchmarkConfig:
    """Configuration settings for remote benchmark execution.

    Encapsulates all environment variables and settings required for
    orchestrating remote GPU benchmarking workflows. All values must
    be provided via environment variables or .env file.
    """

    # Vast.ai configuration
    api_key: str
    gpu_type: str
    num_gpus: int
    disk_space: int

    # Benchmark parameters
    test_iterations: str
    context_start: str
    context_end: str
    context_multiplier: str

    # Ollama configuration
    ollama_model: str
    ollama_flash_attention: str
    ollama_keep_alive: str
    ollama_kv_cache_type: str
    ollama_max_loaded_models: str
    ollama_num_gpu: str
    ollama_num_parallel: str

    # Local storage
    local_results_dir: str

    @classmethod
    def from_dotenv(cls, env_file: str = ".env") -> RemoteBenchmarkConfig:
        """Create configuration from .env file.

        Returns:
            RemoteBenchmarkConfig: An instance of RemoteBenchmarkConfig populated with values
            from the .env file.
        """
        config = dotenv_values(env_file)

        def get_required(key: str) -> str:
            value = config.get(key)
            if value is None:
                msg = f"Missing required environment variable: {key}"
                raise ValueError(msg)
            return value

        def get_required_int(key: str) -> int:
            value = get_required(key)
            try:
                return int(value)
            except ValueError as e:
                msg = f"Invalid integer value for environment variable {key}: {value}"
                raise ValueError(msg) from e

        return cls(
            api_key=get_required("VAST_API_KEY"),
            gpu_type=get_required("GPU_TYPE"),
            num_gpus=get_required_int("NUM_GPUS"),
            disk_space=get_required_int("DISK_SPACE"),
            test_iterations=get_required("TEST_ITERATIONS"),
            context_start=get_required("CONTEXT_START"),
            context_end=get_required("CONTEXT_END"),
            context_multiplier=get_required("CONTEXT_MULTIPLIER"),
            ollama_model=get_required("OLLAMA_MODEL"),
            ollama_flash_attention=get_required("OLLAMA_FLASH_ATTENTION"),
            ollama_keep_alive=get_required("OLLAMA_KEEP_ALIVE"),
            ollama_kv_cache_type=get_required("OLLAMA_KV_CACHE_TYPE"),
            ollama_max_loaded_models=get_required("OLLAMA_MAX_LOADED_MODELS"),
            ollama_num_gpu=get_required("OLLAMA_NUM_GPU"),
            ollama_num_parallel=get_required("OLLAMA_NUM_PARALLEL"),
            local_results_dir=get_required("LOCAL_RESULTS_DIR"),
        )

    @property
    def results_folder_name(self) -> str:
        """Generate standardised results folder name based on configuration.

        Format: {GPU_TYPE}_{NUM_GPUS}x_Ollama-{VERSION}_{OLLAMA_MODEL}
        All non-dot/hyphen special characters are replaced with hyphens.
        """
        # Sanitise GPU type (replace special chars except dots/hyphens with hyphens)
        gpu_clean = self._sanitise_name_component(self.gpu_type)

        # Sanitise model name
        model_clean = self._sanitise_name_component(self.ollama_model)

        # Get Ollama version - will be set by the orchestrator
        ollama_version = getattr(self, "_ollama_version", "unknown")

        return f"{gpu_clean}_{self.num_gpus}x_Ollama-{ollama_version}_{model_clean}"

    def _sanitise_name_component(self, name: str) -> str:
        """Sanitise name component by replacing special chars (except dots/hyphens) with hyphens.

        Returns:
            Sanitised name with special characters replaced by hyphens.
        """
        # Replace any character that's not alphanumeric, dot, or hyphen with hyphen
        return re.sub(r"[^a-zA-Z0-9.-]", "-", name)

    def set_ollama_version(self, version: str) -> None:
        """Set the Ollama version for results folder naming."""
        self._ollama_version = version

    @property
    def remote_results_path(self) -> str:
        """Generate remote results path for benchmark output."""
        return f"/app/results/{self.results_folder_name}"

    @property
    def local_results_path(self) -> Path:
        """Generate local results path for downloaded files."""
        return Path(self.local_results_dir) / self.results_folder_name


# Global configuration defaults from environment variables
HF_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/gemma-3-12b-it-GGUF:Q5_K_XL")
TEST_ITERATIONS = int(os.getenv("TEST_ITERATIONS", "3"))
CONTEXT_START = int(os.getenv("CONTEXT_START", "8192"))
CONTEXT_END = int(os.getenv("CONTEXT_END", "131072"))
CONTEXT_MULTIPLIER = float(os.getenv("CONTEXT_MULTIPLIER", "2"))


@dataclass
class LocalBenchmarkConfig:
    """Configuration settings for local benchmark execution.

    Handles context length generation based on start/end/multiplier parameters
    and scenario configuration for local GPU benchmarking.
    """

    model: str = HF_MODEL
    context_lengths: list[int] | None = None
    runs_per_context: int = TEST_ITERATIONS
    scenarios: list[str] | None = None
    context_start: int = CONTEXT_START
    context_end: int = CONTEXT_END
    context_multiplier: float = CONTEXT_MULTIPLIER
    short_prompt: str = (
        "Write a detailed technical explanation of how neural networks work, "
        "including backpropagation, gradient descent, and modern architectures "
        "like transformers. Include examples and mathematical concepts."
    )
    filler_text: str = (
        "Neural networks represent a fundamental paradigm in artificial intelligence, "
        "drawing inspiration from biological neural systems to create computational "
        "models capable of learning complex patterns from data. These interconnected "
        "networks of artificial neurons have revolutionised machine learning and "
        "enabled breakthroughs in computer vision, natural language processing, "
        "and many other domains."
    )

    def __post_init__(self) -> None:
        """Initialise computed fields after object creation."""
        if self.context_lengths is None:
            self.context_lengths = self._generate_context_lengths()

        if self.scenarios is None:
            self.scenarios = ["short", "half_context"]

    def _generate_context_lengths(self) -> list[int]:
        """Generate context lengths based on start, end, and multiplier.

        Returns:
            List of context lengths following geometric progression.
        """
        lengths = []
        current = self.context_start

        while current <= self.context_end:
            lengths.append(current)
            next_length = int(current * self.context_multiplier)

            if next_length > self.context_end:
                break

            current = next_length

        return lengths


@dataclass
class TestResult:
    """Single benchmark test result with comprehensive GPU and performance metrics.

    Stores all data from one test iteration including Ollama response metrics,
    GPU utilization statistics, and timing information.
    """

    timestamp: str
    scenario: str
    context_length: int
    run_number: int
    tokens_per_second: float
    total_time_seconds: float
    prompt_eval_count: int
    prompt_eval_duration_ms: float
    eval_count: int
    eval_duration_ms: float
    total_duration_ms: float
    load_duration_ms: float
    response_length_chars: int
    gpu_util_avg: float
    gpu_util_max: float
    gpu_util_min: float
    gpu_mem_util_avg: float
    gpu_mem_util_max: float
    gpu_mem_util_min: float
    vram_used_avg_mb: float
    vram_used_max_mb: float
    vram_used_min_mb: float
    vram_free_avg_mb: float
    vram_free_max_mb: float
    vram_free_min_mb: float
    gpu_temp_avg: float
    gpu_temp_max: float
    gpu_temp_min: float
    gpu_power_avg: float
    gpu_power_max: float
    gpu_power_min: float
    gpu_clock_graphics_avg: float
    gpu_clock_memory_avg: float
    gpu_fan_speed_avg: float
    gpu_pstate_mode: str
    test_id: str

#!/usr/bin/env python3
"""Remote GPU LLM Benchmarking Orchestrator.

Automated orchestration tool for executing LLM context length benchmarks on remote
Vast.ai GPU instances. This script handles the complete lifecycle of remote benchmarking:
instance provisioning, Ollama installation and configuration, benchmark execution,
results retrieval, and resource cleanup.

The orchestrator supports configurable GPU types, Ollama versions, and benchmark
parameters through environment variables. It provides comprehensive logging and
error handling for reliable automated benchmarking workflows across different
GPU architectures and configurations.

Key Features:
- Automated Vast.ai instance lifecycle management
- Configurable Ollama installation and model deployment
- Secure file transfer for scripts and results
- Comprehensive error handling and cleanup
- Environment-based configuration management
"""

from __future__ import annotations

from helpers.logger import logger
from helpers.models import RemoteBenchmarkConfig
from helpers.ollama_manager import OllamaManager
from helpers.vastai_api import VastAIDirect, VastAPIError
from helpers.vastai_instance import VastInstance


class BenchmarkExecutor:
    """Executes benchmark scripts on remote instances with proper configuration.

    Coordinates the execution of LLM benchmarks with environment variable
    configuration and result collection.
    """

    def __init__(self, config: RemoteBenchmarkConfig) -> None:
        """Initialise benchmark executor with configuration.

        Args:
            config: Benchmark configuration settings.
        """
        self.config = config

    def run(self, instance: VastInstance) -> None:
        """Execute the benchmark script on the remote instance.

        Args:
            instance: Target instance for benchmark execution.
        """
        # Export environment variables individually for clarity and debugging
        # Commands to set up environment variables and change directory
        setup_commands = (
            f"set -e && "  # Exit on any error
            f"echo '🔍 Starting benchmark command execution...' && "
            f"cd /app && "
            f"echo '📁 Changed to /app directory' && "
            f"export PYTHONUNBUFFERED=1 && "
            f"export TEST_ITERATIONS={self.config.test_iterations} && "
            f"export CONTEXT_START={self.config.context_start} && "
            f"export CONTEXT_END={self.config.context_end} && "
            f"export CONTEXT_MULTIPLIER={self.config.context_multiplier} && "
            f"export OLLAMA_MODEL='{self.config.ollama_model}' && "
            f"export OLLAMA_FLASH_ATTENTION={self.config.ollama_flash_attention} && "
            f"export OLLAMA_KEEP_ALIVE={self.config.ollama_keep_alive} && "
            f"export OLLAMA_KV_CACHE_TYPE={self.config.ollama_kv_cache_type} && "
            f"export OLLAMA_MAX_LOADED_MODELS={self.config.ollama_max_loaded_models} && "
            f"export OLLAMA_NUM_GPU={self.config.ollama_num_gpu} && "
            f"export OLLAMA_NUM_PARALLEL={self.config.ollama_num_parallel} && "
            f"export OUTPUT_PATH={self.config.remote_results_path} && "
            f"echo '⚙️ Environment variables exported'"
        )

        # Command to run the benchmark script
        benchmark_script_command = (
            "set -e && "  # Exit on any error
            "cd /app && "
            "echo '🚀 Starting benchmark script...' && "
            "uv run scripts/llm_benchmark.py"
        )

        logger.info("⚡ Starting setup commands execution...")
        logger.debug("Setup command: %s", setup_commands)
        instance.execute_command_streaming(setup_commands, timeout=60)  # Short timeout for setup

        logger.info("⚡ Starting benchmark script execution...")
        logger.debug("Benchmark script command: %s", benchmark_script_command)
        instance.execute_command_streaming(benchmark_script_command, timeout=1800)  # 30 minutes


class RemoteBenchmarkOrchestrator:
    """Main orchestrator for remote GPU benchmarking workflows.

    Coordinates the complete lifecycle of remote benchmarking from instance
    provisioning through results retrieval and cleanup. Follows Single
    Responsibility Principle by delegating specific tasks to specialised classes.
    """

    def __init__(self, config: RemoteBenchmarkConfig) -> None:
        """Initialise orchestrator with configuration.

        Args:
            config: Complete benchmark configuration.
        """
        self.config = config
        self.client = VastAIDirect(api_key=config.api_key)
        self.ollama_manager = OllamaManager(config)
        self.benchmark_executor = BenchmarkExecutor(config)

    def run(self) -> None:
        """Execute the complete remote benchmarking workflow.

        Orchestrates all stages with proper error handling and cleanup.

        Raises:
            SystemExit: If any critical operation fails.
        """
        logger.info("🚀 Starting remote benchmark process...")

        # Prepare local environment
        self.config.local_results_path.mkdir(parents=True, exist_ok=True)

        # Launch and configure instance
        instance = VastInstance(self.client, self.config)

        try:
            instance.launch()

            # Setup Ollama environment and get version for results naming
            ollama_version = self.ollama_manager.start_and_configure(instance)
            self.config.set_ollama_version(ollama_version)

            # Execute benchmark
            self.benchmark_executor.run(instance)

            # Retrieve results
            instance.download_results(
                self.config.remote_results_path, self.config.local_results_path
            )

        except VastAPIError as e:
            # Handle known API errors gracefully without traceback spam
            logger.error("Benchmark workflow failed: %s", str(e))
            raise SystemExit(1) from e
        except Exception as e:
            # Unknown errors get full traceback for debugging
            logger.exception("Benchmark workflow failed with unexpected error")
            raise SystemExit(1) from e
        finally:
            # Always cleanup instance
            instance.destroy()

        logger.info("Process finished successfully.")


def main() -> None:
    """Main entry point for remote benchmarking orchestration.

    Creates configuration from environment and executes the complete workflow.

    Raises:
        SystemExit: If any critical operation fails.
    """
    try:
        config = RemoteBenchmarkConfig.from_dotenv()
        orchestrator = RemoteBenchmarkOrchestrator(config)
        orchestrator.run()
    except (FileNotFoundError, ValueError) as e:
        logger.exception("Configuration error")
        raise SystemExit(1) from e
    except Exception as e:
        logger.exception("Unexpected error in main")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()

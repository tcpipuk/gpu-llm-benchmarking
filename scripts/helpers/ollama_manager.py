"""Ollama service management for remote instances.

This module provides functionality for managing Ollama service startup
and configuration on remote GPU instances for benchmarking workflows.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from .logger import logger

if TYPE_CHECKING:
    from typing import Any

    from .models import RemoteBenchmarkConfig

    VastInstance = Any


class OllamaManager:
    """Manages Ollama service startup and configuration on remote instances.

    Handles the startup of the pre-installed Ollama service and model deployment
    for consistent benchmarking environments.
    """

    def __init__(self, config: RemoteBenchmarkConfig) -> None:
        """Initialise Ollama manager with configuration.

        Args:
            config: Benchmark configuration containing Ollama settings.
        """
        self.config = config

    def start_and_configure(self, instance: VastInstance) -> str:
        """Start Ollama service and configure environment on remote instance.

        Args:
            instance: Target Vast.ai instance for configuration.

        Returns:
            Ollama version string for results folder naming.
        """
        # Start Ollama service with environment variables in background
        logger.info("⚡ Starting Ollama server in background")
        start_command = (
            f"cd /app && "
            f"OLLAMA_FLASH_ATTENTION={self.config.ollama_flash_attention} "
            f"OLLAMA_KEEP_ALIVE={self.config.ollama_keep_alive} "
            f"OLLAMA_KV_CACHE_TYPE={self.config.ollama_kv_cache_type} "
            f"OLLAMA_MAX_LOADED_MODELS={self.config.ollama_max_loaded_models} "
            f"OLLAMA_NUM_GPU={self.config.ollama_num_gpu} "
            f"OLLAMA_NUM_PARALLEL={self.config.ollama_num_parallel} "
            f"nohup ollama serve > ollama.log 2>&1 & disown"
        )

        instance.execute_command_background(start_command)

        # Wait for Ollama server to be ready and get version
        version = self._get_ollama_version(instance)

        # Pull the model with streaming output for real-time progress monitoring
        logger.info("📦 Pulling model %s", self.config.ollama_model)
        pull_command = f"ollama pull {self.config.ollama_model}"
        instance.execute_command_streaming(
            pull_command,
            timeout=1200,  # 20 minutes for large models
        )

        # Test generation to ensure model is loaded into VRAM
        logger.info("🎯 Running test generation to load model into VRAM")
        test_command = (
            f"curl -X POST http://localhost:11434/api/generate "
            f'-H "Content-Type: application/json" '
            f'-d \'{{"model": "{self.config.ollama_model}", '
            f'"prompt": "Say hello", '
            f'"stream": false, "options": {{"num_predict": 10}}}}\''
        )
        response = instance.execute_command_capture(test_command, timeout=60)
        try:
            response_data = json.loads(response)
            actual_response = response_data.get("response", "No response field found")
            logger.debug("Test generation response: %r", actual_response)
        except (json.JSONDecodeError, AttributeError):
            # Truncate response for debug logging
            max_debug_length = 100
            truncated_response = (
                response[:max_debug_length] + "..."
                if len(response) > max_debug_length
                else response
            )
            logger.debug("Test generation response (raw): %s", truncated_response)
        logger.info("🎉 Test generation complete - model should now be in VRAM")

        logger.info("🚀 Ollama ready for testing with model preloaded")
        return version

    def _get_ollama_version(self, instance: VastInstance) -> str:
        """Wait for Ollama server to be ready and get its version.

        Args:
            instance: The remote instance to check.

        Returns:
            The Ollama version string.

        Raises:
            RuntimeError: If the server doesn't become ready within the timeout.
        """
        logger.info("⏳ Waiting for Ollama server...")
        max_attempts = 12  # 60 seconds total (5 second intervals)
        debug_threshold = 2  # After 10 seconds (2 attempts), check logs

        for attempt in range(max_attempts):
            try:
                version_output = str(
                    instance.execute_command_capture("ollama --version", timeout=10)
                )
                if "version is" in version_output:
                    version = version_output.rsplit("version is", maxsplit=1)[-1].strip()
                    logger.info("✅ Ollama server ready, version: %s", version)
                    return version
            except RuntimeError as e:
                logger.debug(
                    "Ollama not ready yet (attempt %d/%d): %s", attempt + 1, max_attempts, e
                )

                # After 10 seconds, check the ollama.log for debugging
                if attempt == debug_threshold - 1:
                    logger.warning("Ollama not responding after 10 seconds, checking logs...")
                    try:
                        log_output = instance.execute_command_capture(
                            "cd /app && cat ollama.log", timeout=5
                        )
                        logger.debug("Ollama log contents:\n%s", log_output)
                    except Exception as log_error:
                        logger.debug("Could not read ollama.log: %s", log_error)

            logger.debug("Waiting 5 seconds before next attempt...")
            time.sleep(5)

        # Final attempt to get logs before failing
        logger.error("Ollama failed to start. Checking final log state...")
        try:
            final_log = instance.execute_command_capture("cd /app && cat ollama.log", timeout=5)
            logger.error("Final ollama.log contents:\n%s", final_log)
        except Exception as e:
            logger.error("Could not read final ollama.log: %s", e)

        msg = "Ollama server failed to become ready after 60 seconds."
        raise RuntimeError(msg)

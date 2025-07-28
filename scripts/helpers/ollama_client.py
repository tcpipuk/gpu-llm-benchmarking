"""Ollama API client for GPU benchmarking.

This module provides a simple Ollama API client for interacting with the
Ollama service during benchmark execution.
"""

from __future__ import annotations

from typing import Any

import requests

from helpers.logger import logger


class OllamaClient:
    """Simple Ollama API client."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        """Initialise Ollama client with base URL and session."""
        self.base_url = base_url
        self.session = requests.Session()

    def generate(
        self, model: str, prompt: str, context_length: int, timeout: int = 60
    ) -> dict[str, Any]:
        """Generate text using Ollama API.

        Returns:
            Ollama response as JSON.

        Raises:
            requests.Timeout: If request times out.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": context_length},
        }
        url = f"{self.base_url}/api/generate"

        logger.debug("📤 Sending POST request to %s", url)
        logger.debug(
            "📊 Payload: model=%s, prompt_length=%d, num_ctx=%d, timeout=%ds",
            model,
            len(prompt),
            context_length,
            timeout,
        )

        try:
            response = self.session.post(url, json=payload, timeout=timeout)
            logger.debug("📥 Response status: %d", response.status_code)
            response.raise_for_status()

            result = response.json() or {}
            logger.debug(
                "📋 Response stats: eval_count=%d, total_duration=%dms",
                result.get("eval_count", 0),
                result.get("total_duration", 0) // 1_000_000,
            )
        except requests.Timeout:
            logger.error("⏱️ Ollama request timed out after %d seconds", timeout)
            logger.error(
                "   Model: %s, Context: %d, Prompt length: %d", model, context_length, len(prompt)
            )
            raise
        else:
            return result

"""Ollama API client for GPU benchmarking.

This module provides a simple Ollama API client for interacting with the
Ollama service during benchmark execution.
"""

from __future__ import annotations

from typing import Any

import requests


class OllamaClient:
    """Simple Ollama API client."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        """Initialise Ollama client with base URL and session."""
        self.base_url = base_url
        self.session = requests.Session()

    def generate(self, model: str, prompt: str, context_length: int) -> dict[str, Any]:
        """Generate text using Ollama API.

        Returns:
            Ollama response as JSON.
        """
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": context_length},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json() or {}

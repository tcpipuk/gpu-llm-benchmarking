"""Prompt generation utilities for GPU benchmarking.

This module provides the PromptGenerator class for creating test prompts
of different scenarios and context lengths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import LocalBenchmarkConfig


class PromptGenerator:
    """Generates prompts for different test scenarios.

    Creates short prompts for allocation overhead testing and longer prompts
    proportional to context length for realistic usage testing.
    """

    @staticmethod
    def generate(scenario: str, context_length: int, config: LocalBenchmarkConfig) -> str:
        """Generate prompt based on scenario and context length.

        Args:
            scenario: 'short' or 'half_context'
            context_length: Target context window size
            config: Benchmark configuration with prompt templates

        Returns:
            Generated prompt string appropriate for the scenario.
        """
        if scenario == "short":
            return config.short_prompt

        # Half-context scenario
        target_tokens = context_length // 2
        target_chars = target_tokens * 4  # ~4 chars per token

        prompt_parts = [config.short_prompt, "\n\nContext: "]
        current_length = len("".join(prompt_parts))

        # Add filler to reach target length
        while current_length < target_chars:
            prompt_parts.append(config.filler_text + " ")
            current_length = len("".join(prompt_parts))

        prompt_parts.append("\n\nGiven this context, provide a comprehensive response.")
        return "".join(prompt_parts)

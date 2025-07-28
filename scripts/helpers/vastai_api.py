"""Direct API client for Vast.ai operations.

This module provides a direct API client for interacting with Vast.ai services,
including instance management, offer searching, and command execution.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from .logger import logger


class VastAPIError(Exception):
    """Custom exception for known Vast.ai API errors that should be handled gracefully."""

    def __init__(self, message: str, error_type: str = "unknown") -> None:
        """Initialise VastAPIError with message and error type.

        Args:
            message: The user-friendly error message.
            error_type: The type of error for categorisation.
        """
        super().__init__(message)
        self.error_type = error_type


class VastAIDirect:
    """Direct API implementation for Vast.ai operations."""

    def __init__(self, api_key: str) -> None:
        """Initialise Vast.ai direct API client."""
        self.api_key = api_key
        self.base_url = "https://console.vast.ai/api/v0"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout = 30

    def search_offers(
        self, gpu_name: str, num_gpus: int = 1, disk_space: float = 30.0
    ) -> list[dict[str, Any]]:
        """Search for available GPU offers.

        Returns:
            List of available GPU offers.
        """
        params: dict[str, str | int] = {
            "q": json.dumps({
                "gpu_name": {"eq": gpu_name},
                "num_gpus": {"eq": num_gpus},
                "disk_space": {"gte": disk_space},
                "rentable": {"eq": True},
            }),
            "order": "score-",
            "limit": 10,
        }

        logger.debug("[API] GET %s/bundles - Searching for GPU offers", self.base_url)
        response = requests.get(
            f"{self.base_url}/bundles", headers=self.headers, params=params, timeout=self.timeout
        )

        if not response.ok:
            logger.error("Search failed: %s - %s", response.status_code, response.text)
            return []

        data = response.json()
        if isinstance(data, list):
            return data
        return list(data.get("offers", []))

    def create_instance(
        self, offer_id: int | None, image: str, disk: float = 30.0, onstart_cmd: str | None = None
    ) -> dict[str, Any]:
        """Create an instance from an offer.

        Returns:
            Dictionary containing the instance details.
        """
        payload = {
            "id": offer_id,
            "image": image,
            "disk": disk,
            "ssh": True,
        }

        if onstart_cmd:
            payload["onstart_cmd"] = onstart_cmd

        logger.debug(
            "[API] PUT %s/asks/%s/ - Creating instance from offer", self.base_url, offer_id
        )
        response = requests.put(
            f"{self.base_url}/asks/{offer_id}/", headers=self.headers, json=payload, timeout=60
        )

        if not response.ok:
            # Debug log the raw API response for troubleshooting
            logger.debug("Raw API error response: %s", response.text)
            error_msg = self._parse_api_error(response)
            return {"error": error_msg, "status_code": response.status_code}

        data: dict[str, Any] = response.json()
        return data

    def _parse_api_error(self, response: requests.Response) -> str:
        """Parse API error response and provide user-friendly messages.

        Args:
            response: The failed HTTP response.

        Returns:
            A user-friendly error message.
        """
        try:
            error_data = response.json()
            error_type = error_data.get("error", "unknown_error")
            error_msg = error_data.get(
                "msg", error_data.get("message", "No error message provided")
            )

            # Handle specific error types with helpful guidance
            if error_type == "insufficient_credit":
                return (
                    f"❌ Insufficient credit in your Vast.ai account.\n"
                    f"💡 Please add credit to your account at: https://console.vast.ai/billing/\n"
                    f"   Original message: {error_msg}"
                )
            if error_type == "bad_request":
                return (
                    f"❌ Invalid request parameters.\n"
                    f"💡 Check your GPU configuration and ensure the instance is still available.\n"
                    f"   Original message: {error_msg}"
                )
            if error_type == "offer_unavailable":
                return (
                    f"❌ The selected GPU offer is no longer available.\n"
                    f"💡 Try running the script again to find new offers.\n"
                    f"   Original message: {error_msg}"
                )
        except (json.JSONDecodeError, AttributeError):
            # Fallback if response isn't valid JSON
            return f"❌ HTTP {response.status_code}: {response.text}"
        else:
            return f"❌ API Error ({error_type}): {error_msg}"

    def get_instance_status(self, instance_id: str) -> dict[str, Any]:
        """Get status of an instance.

        Returns:
            Dictionary containing the instance status.
        """
        logger.debug(
            "[API] GET %s/instances/%s/ - Getting instance status", self.base_url, instance_id
        )
        response = requests.get(
            f"{self.base_url}/instances/{instance_id}/", headers=self.headers, timeout=self.timeout
        )

        if not response.ok:
            return {"error": response.text, "status_code": response.status_code}

        data: dict[str, Any] = response.json()
        return data

    def get_ssh_connection(self, instance_id: str) -> dict[str, Any]:
        """Get SSH connection details for an instance.

        Returns:
            Dictionary containing SSH connection details including host and port.
        """
        logger.debug(
            "[API] GET %s/instances/%s/ssh/ - Getting SSH connection details",
            self.base_url,
            instance_id,
        )
        response = requests.get(
            f"{self.base_url}/instances/{instance_id}/ssh/",
            headers=self.headers,
            timeout=self.timeout,
        )

        if not response.ok:
            return {"error": response.text, "status_code": response.status_code}

        data: dict[str, Any] = response.json()
        return data

    def destroy_instance(self, instance_id: str) -> bool:
        """Destroy an instance.

        Returns:
            True if the instance was destroyed successfully, False otherwise.
        """
        logger.debug(
            "[API] DELETE %s/instances/%s/ - Destroying instance", self.base_url, instance_id
        )
        response = requests.delete(
            f"{self.base_url}/instances/{instance_id}/", headers=self.headers, timeout=self.timeout
        )

        return response.ok

    def execute_command(self, instance_id: str, command: str) -> dict[str, Any]:
        """Execute a command on an instance.

        Returns:
            Dictionary containing the command result.
        """
        logger.debug(
            "[API] POST %s/instances/%s/execute/ - Executing command on instance",
            self.base_url,
            instance_id,
        )
        response = requests.post(
            f"{self.base_url}/instances/{instance_id}/execute/",
            headers=self.headers,
            json={"command": command},
            timeout=120,
        )

        if not response.ok:
            return {"error": response.text, "status_code": response.status_code}

        try:
            data: dict[str, Any] = response.json()
        except json.JSONDecodeError:
            # Handle cases where response isn't valid JSON
            return {"error": "Invalid JSON response", "raw_response": response.text}
        else:
            return data

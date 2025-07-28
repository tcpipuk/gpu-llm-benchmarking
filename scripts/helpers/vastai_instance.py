"""Vast.ai instance management for remote GPU benchmarking.

This module provides the VastInstance class that handles the complete lifecycle
of Vast.ai GPU instances including provisioning, status monitoring, SSH connectivity,
and cleanup operations for benchmarking workflows.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import requests

from .interactive import SSHInteractiveSession
from .logger import logger
from .vastai_api import VastAIDirect, VastAPIError

if TYPE_CHECKING:
    from pathlib import Path

    from .models import RemoteBenchmarkConfig


class VastInstance:
    """Represents a Vast.ai GPU instance with lifecycle management."""

    def __init__(self, client: VastAIDirect, config: RemoteBenchmarkConfig) -> None:
        """Initialise Vast.ai instance manager."""
        self.client = client
        self.config = config
        self.instance_id: str | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self.ssh_user: str = "root"
        self.ssh_session: SSHInteractiveSession | None = None

    def launch(self) -> None:
        """Launch and configure a new GPU instance."""
        logger.info("🚀 Launching %sx %s instance...", self.config.num_gpus, self.config.gpu_type)
        self._create_instance_with_retry()
        instance_data = self._wait_for_readiness()
        self._setup_ssh_connection(instance_data)

    def _create_instance_with_retry(self) -> None:
        """Create instance with retry logic for offer availability.

        Raises:
            RuntimeError: If the instance cannot be created.
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            logger.info(
                "✨ Searching for available offers (attempt %d/%d)...", attempt + 1, max_attempts
            )
            offer = self._find_best_offer()
            if self._try_create_instance(offer, attempt, max_attempts):
                return
        msg = f"Failed to launch instance after {max_attempts} attempts"
        raise RuntimeError(msg)

    def _find_best_offer(self) -> dict[str, Any]:
        """Find the best available offer for the configured GPU type.

        Returns:
            The best offer found.
        """
        offers = self.client.search_offers(
            gpu_name=self.config.gpu_type,
            num_gpus=self.config.num_gpus,
            disk_space=float(self.config.disk_space),
        )
        if not offers:
            self._raise_no_offers_error()
        logger.info("🎯 Found %s offers, trying best offer", len(offers))
        return offers[0]

    def _raise_no_offers_error(self) -> None:
        """Raise informative error when no offers are available.

        Raises:
            RuntimeError: Always, with a descriptive message.
        """
        logger.info("No offers found, checking available GPU types...")
        available_gpus = self._get_available_gpu_types()
        msg = f"No verified offers found for {self.config.num_gpus}x {self.config.gpu_type}"
        if available_gpus:
            msg += f"\n\nAvailable GPU types: {', '.join(sorted(available_gpus))}"
            msg += "\n\nPlease update GPU_TYPE in your .env file to one of the available types."
        raise RuntimeError(msg)

    def _try_create_instance(self, offer: dict[str, Any], attempt: int, max_attempts: int) -> bool:
        """Try to create instance from offer.

        Returns:
            True if the instance was created successfully, False otherwise.

        Raises:
            RuntimeError: If instance creation fails after all retries.
            VastAPIError: If there is an API-level error from Vast.ai.
        """
        offer_id = offer.get("id")
        if offer_id is None:
            logger.warning("Offer has no ID, skipping")
            return False

        logger.info(
            "💰 Selected offer: ID=%s, Price=$%.3f/hr, Location=%s",
            offer_id,
            offer.get("dph_total", 0),
            offer.get("geolocation", "Unknown"),
        )
        try:
            result = self.client.create_instance(
                offer_id=offer_id,
                image="ghcr.io/tcpipuk/gpu-llm-benchmarking:latest",
                disk=float(self.config.disk_space),
            )
            if "error" in result:
                return self._handle_creation_error(
                    result["error"], offer, offer_id, attempt, max_attempts
                )
            self.instance_id = str(result.get("new_contract"))
            logger.info("✅ Instance %s created successfully.", self.instance_id)
        except VastAPIError:
            raise
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning("Instance creation failed, retrying: %s", e)
                time.sleep(2)
                return False
            msg = "Failed to launch instance after all retry attempts"
            raise RuntimeError(msg) from e
        else:
            return True

    def _handle_creation_error(
        self, error_msg: str, offer: dict[str, Any], offer_id: str, attempt: int, max_attempts: int
    ) -> bool:
        """Handle instance creation errors with appropriate retry logic.

        Returns:
            False if the error can be retried, otherwise raises an exception.

        Raises:
            RuntimeError: For unrecoverable errors.
            VastAPIError: For specific API errors like insufficient credit.
        """
        if "credit" in error_msg.lower() and "insufficient" in error_msg.lower():
            msg = (
                f"💳 Cannot launch GPU instance due to insufficient credit.\n\n"
                f"Selected offer details:\n"
                f"  • GPU: {self.config.num_gpus}x {self.config.gpu_type}\n"
                f"  • Price: ${offer.get('dph_total', 0):.3f}/hour\n"
                f"  • Location: {offer.get('geolocation', 'Unknown')}\n\n"
                f"Please add credit to your Vast.ai account and try again.\n"
                f"💡 Billing page: https://console.vast.ai/billing/"
            )
            raise VastAPIError(msg, "insufficient_credit")

        if "no_such_ask" in error_msg.lower() and attempt < max_attempts - 1:
            logger.warning("Offer %s no longer available, searching for new offers...", offer_id)
            time.sleep(2)
            return False

        if any(
            known_error in error_msg.lower() for known_error in ["bad_request", "offer_unavailable"]
        ):
            raise VastAPIError(error_msg, "api_error")

        msg = f"Failed to create instance: {error_msg}"
        raise RuntimeError(msg)

    def _wait_for_readiness(self) -> dict[str, Any]:
        """Wait for instance to become ready for use.

        Returns:
            The instance data when ready.

        Raises:
            RuntimeError: If the instance fails to become ready within the timeout.
        """
        logger.info("⏳ Waiting for instance to become ready...")
        max_wait_time = 600
        start_time = time.time()
        last_status = None

        # Initial delay to allow instance to start up before first API check
        time.sleep(20)

        while time.time() - start_time < max_wait_time:
            try:
                if self.instance_id is None:
                    msg = "Instance ID is None"
                    raise RuntimeError(msg)
                status = self.client.get_instance_status(self.instance_id)
                instance_data: dict[str, Any] = status.get("instances", status)
                actual_status = instance_data.get("actual_status")

                if actual_status != last_status:
                    if actual_status is None:
                        logger.info("⏳ Instance initialising...")
                    elif actual_status == "loading":
                        logger.info("🐳 Loading Docker image...")
                    elif actual_status == "running":
                        logger.info("🌟 Instance is running")
                        self._log_instance_metadata(instance_data)
                        return instance_data
                    else:
                        logger.info("Instance status: %s", actual_status)
                    last_status = actual_status

                if "error" not in status and actual_status == "running":
                    return instance_data
            except Exception as e:
                logger.warning("Error checking instance status: %s", e)

            time.sleep(20)

        self.destroy()
        msg = "Instance failed to become ready within timeout"
        raise RuntimeError(msg)

    def _log_instance_metadata(self, instance_data: dict[str, Any]) -> None:
        """Log useful instance metadata for analysis and debugging."""
        metadata = {
            "gpu_name": instance_data.get("gpu_name"),
            "gpu_ram": instance_data.get("gpu_ram"),
            "cpu_name": instance_data.get("cpu_name"),
            "cpu_cores": instance_data.get("cpu_cores"),
            "cpu_ram": instance_data.get("cpu_ram"),
            "location": instance_data.get("geolocation"),
            "host_id": instance_data.get("host_id"),
            "machine_id": instance_data.get("machine_id"),
            "driver_version": instance_data.get("driver_version"),
            "cost_per_hour": instance_data.get("dph_total"),
            "disk_name": instance_data.get("disk_name"),
            "disk_bw": instance_data.get("disk_bw"),
            "compute_cap": instance_data.get("compute_cap"),
            "pcie_bw": instance_data.get("pcie_bw"),
            "reliability": instance_data.get("reliability2"),
        }
        logger.info("💻 Instance Details:")
        logger.info("   GPU: %s (%s MB VRAM)", metadata["gpu_name"], metadata["gpu_ram"])
        logger.info(
            "   CPU: %s (%s cores, %s MB RAM)",
            metadata["cpu_name"],
            metadata["cpu_cores"],
            metadata["cpu_ram"],
        )
        logger.info(
            "   Location: %s (Host: %s, Machine: %s)",
            metadata["location"],
            metadata["host_id"],
            metadata["machine_id"],
        )
        logger.info(
            "   Cost: $%.3f/hour | Driver: %s | Reliability: %.1f%%",
            metadata["cost_per_hour"] or 0,
            metadata["driver_version"],
            (metadata["reliability"] or 0) * 100,
        )
        logger.info(
            "   Storage: %s (%.1f MB/s) | Compute: %s | PCIe: %.1f GB/s",
            metadata["disk_name"],
            metadata["disk_bw"] or 0,
            metadata["compute_cap"],
            metadata["pcie_bw"] or 0,
        )

    def _setup_ssh_connection(self, instance_data: dict[str, Any] | None = None) -> None:
        """Extract and store SSH connection details and test connectivity.

        Args:
            instance_data: Optional instance data from previous API call to avoid
                redundant requests.

        Raises:
            RuntimeError: If SSH details cannot be retrieved.
        """
        logger.info("🔐 Getting SSH connection details...")
        try:
            if self.instance_id is None:
                msg = "Instance ID is None"
                raise RuntimeError(msg)

            # Use provided instance data or fetch it
            if instance_data is None:
                status = self.client.get_instance_status(self.instance_id)
                if "error" in status:
                    msg = f"Failed to get instance status: {status}"
                    raise RuntimeError(msg)
                instance_data = status.get("instances", status)

            self.ssh_host = instance_data.get("ssh_host") or instance_data.get("public_ipaddr")
            self.ssh_port = int(instance_data.get("ssh_port", 22))

            if not self.ssh_host:
                msg = "No SSH host found in instance status"
                raise RuntimeError(msg)

            logger.info("💻 SSH connection: %s@%s:%s", self.ssh_user, self.ssh_host, self.ssh_port)
            self._wait_for_ssh_ready()

        except Exception as e:
            msg = "Failed to get SSH connection details"
            raise RuntimeError(msg) from e

    def _wait_for_ssh_ready(self) -> None:
        """Wait for SSH to become available and establish a persistent session.

        Raises:
            RuntimeError: If the SSH connection fails.
        """
        logger.info("🔑 Waiting for SSH access...")
        if not self.ssh_host or not self.ssh_port:
            msg = "SSH host/port not set."
            raise RuntimeError(msg)

        self.ssh_session = SSHInteractiveSession(self.ssh_host, self.ssh_port, self.ssh_user)

        # Retry SSH connection with exponential backoff
        max_retries = 10
        base_delay = 2
        for attempt in range(max_retries):
            try:
                self.ssh_session.connect()
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "SSH connection attempt %d/%d failed: %s. Retrying in %d seconds...",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    self.destroy()
                    msg = f"SSH connection failed after {max_retries} attempts."
                    raise RuntimeError(msg) from e
            else:
                logger.info("✅ SSH connection established")
                return

    def transfer_file(self, local_path: str, remote_path: str) -> None:
        """Transfer a file to the remote instance using SFTP.

        Raises:
            RuntimeError: If the file transfer fails.
        """
        if not self.ssh_session or not self.ssh_session.sftp:
            msg = "SFTP session not established."
            raise RuntimeError(msg)

        logger.info("Transferring %s to instance...", local_path)
        try:
            self.ssh_session.sftp.put(local_path, remote_path)
            logger.info("File transferred successfully.")
        except Exception as e:
            self.destroy()
            msg = f"SFTP upload failed: {e}"
            raise RuntimeError(msg) from e

    def download_results(self, remote_path: str, local_path: Path) -> None:
        """Download benchmark results from remote instance using SFTP.".

        Raises:
            RuntimeError: If the SFTP session is not established.
        """
        if not self.ssh_session or not self.ssh_session.sftp:
            msg = "SFTP session not established."
            raise RuntimeError(msg)

        local_path.mkdir(parents=True, exist_ok=True)
        logger.info("Transferring results to %s...", local_path)
        try:
            # List contents of the remote directory
            remote_files = self.ssh_session.sftp.listdir(remote_path)
            for filename in remote_files:
                remote_filepath = f"{remote_path}/{filename}"
                local_filepath = local_path / filename
                self.ssh_session.sftp.get(remote_filepath, str(local_filepath))
            logger.info("Results transferred successfully.")
        except Exception as e:
            logger.error("SFTP download failed: %s", e)

    def execute_command_interactive(
        self, command: str, _description: str = "", timeout: int = 600
    ) -> None:
        """Execute a command interactively via the persistent SSH session.

        Raises:
            RuntimeError: If the SSH session is not established.
        """
        if not self.ssh_session:
            msg = "SSH session not established"
            raise RuntimeError(msg)
        try:
            self.ssh_session.execute_command_interactive(command, timeout)
        except Exception:
            self.destroy()
            raise

    def execute_command_capture(self, command: str, timeout: int = 30) -> str:
        """Execute a command and capture its output via the persistent SSH session.

        Returns:
            The command output.

        Raises:
            RuntimeError: If the SSH session is not established.
        """
        if not self.ssh_session:
            msg = "SSH session not established"
            raise RuntimeError(msg)
        try:
            return self.ssh_session.execute_command_capture(command, timeout)
        except Exception:
            self.destroy()
            raise

    def execute_command_background(self, command: str) -> None:
        """Execute a command in the background without waiting for output.

        This method is designed for fire-and-forget commands that run in the
        background (e.g., using nohup, &, or disown).

        Raises:
            RuntimeError: If the SSH session is not established.
        """
        if not self.ssh_session:
            msg = "SSH session not established"
            raise RuntimeError(msg)
        try:
            self.ssh_session.execute_command_background(command)
        except Exception:
            self.destroy()
            raise

    def execute_command_streaming(self, command: str, timeout: int = 600) -> None:
        """Execute a command with real-time output streaming.

        This method provides real-time output streaming for long-running commands
        like model downloads, showing progress as it happens.

        Raises:
            RuntimeError: If the SSH session is not established.
        """
        if not self.ssh_session:
            msg = "SSH session not established"
            raise RuntimeError(msg)
        try:
            self.ssh_session.execute_command_streaming(command, timeout)
        except Exception:
            self.destroy()
            raise

    def _get_available_gpu_types(self) -> list[str]:
        """Get a list of available GPU types from Vast.ai offers.

        Returns:
            A list of available GPU types.
        """
        try:
            params: dict[str, str | int] = {
                "q": json.dumps({"rentable": {"eq": True}}),
                "order": "score-",
                "limit": 100,
            }
            response = requests.get(
                f"{self.client.base_url}/bundles",
                headers=self.client.headers,
                params=params,
                timeout=self.client.timeout,
            )
            if not response.ok:
                logger.warning("Failed to fetch available GPU types: %s", response.status_code)
                return []
            data = response.json()
            offers = data if isinstance(data, list) else data.get("offers", [])
            gpu_types = {offer.get("gpu_name") for offer in offers if offer.get("gpu_name")}
            return sorted(gpu_types)
        except Exception as e:
            logger.warning("Error fetching available GPU types: %s", e)
            return []

    def destroy(self) -> None:
        """Clean up the instance and close the SSH session."""
        if self.ssh_session:
            self.ssh_session.close()
            self.ssh_session = None

        if not self.instance_id:
            return

        logger.info("Destroying instance...")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.client.destroy_instance(self.instance_id)
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning("Destroy attempt %d failed, retrying: %s", attempt + 1, e)
                    time.sleep(5)
                else:
                    logger.error(
                        "Failed to destroy instance after %d attempts: %s", max_attempts, e
                    )
                    logger.warning(
                        "⚠️  Instance %s may still be running and incurring charges!",
                        self.instance_id,
                    )
            else:
                logger.info("Instance %s destroyed successfully.", self.instance_id)
                return

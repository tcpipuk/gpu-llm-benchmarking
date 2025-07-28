"""Interactive SSH session management for remote command execution.

This module provides functionality for executing commands on remote instances
using pexpect for real-time output streaming and interactive session management.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import paramiko

if TYPE_CHECKING:
    from paramiko import ChannelFile

logger = logging.getLogger(__name__)


class SSHInteractiveSession:
    """Manages SSH sessions for remote command execution and file transfer."""

    def __init__(self, ssh_host: str, ssh_port: int, ssh_user: str = "root") -> None:
        """Initialise SSH session parameters."""
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.client: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None

    def _clean_output(self, text: str) -> str:
        """Returns the text as-is without any filtering.

        Returns:
            The unmodified text.
        """
        return text

    def connect(self, timeout: int = 30) -> None:
        """Establish SSH connection using Paramiko.

        Raises:
            RuntimeError: If the connection fails.
        """
        if self.client:
            transport = self.client.get_transport()
            if transport and transport.is_active():
                logger.debug("SSH session already connected.")
                return

        logger.debug(
            "Establishing SSH connection to %s@%s:%s", self.ssh_user, self.ssh_host, self.ssh_port
        )
        try:
            self.client = paramiko.SSHClient()
            self.client.load_system_host_keys()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(
                hostname=self.ssh_host,
                port=self.ssh_port,
                username=self.ssh_user,
                timeout=timeout,
                auth_timeout=timeout,
                banner_timeout=timeout,
            )
            # Enable keepalive to detect broken connections
            transport = self.client.get_transport()
            if transport:
                transport.set_keepalive(30)  # Send keepalive every 30 seconds
            self.sftp = self.client.open_sftp()
            logger.debug("SSH connection established.")
        except paramiko.AuthenticationException as e:
            msg = f"Authentication failed: {e}"
            raise RuntimeError(msg) from e
        except paramiko.SSHException as e:
            msg = f"SSH connection failed: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"Failed to connect: {e}"
            raise RuntimeError(msg) from e

    def execute_command_interactive(self, command: str, timeout: int = 600) -> None:
        """Execute a command with real-time output.

        Raises:
            RuntimeError: If the command fails.
        """
        if not self.client:
            self.connect()
        if not self.client:
            msg = "SSH session is not properly initialised."
            raise RuntimeError(msg)

        # Check if connection is still alive
        transport = self.client.get_transport()
        if not transport or not transport.is_active():
            logger.warning("SSH connection lost, reconnecting...")
            self.close()
            self.connect()

        logger.debug("Executing command: %s", command)
        _stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        output = stdout.read().decode("utf-8")
        error_output = stderr.read().decode("utf-8")
        exit_status = stdout.channel.recv_exit_status()

        # Debug log sanitised SSH response
        sanitised_output = self._clean_output(output)
        sanitised_error = self._clean_output(error_output)
        logger.debug(
            "SSH Response [exit=%d]: stdout=%r stderr=%r",
            exit_status,
            sanitised_output[:500],
            sanitised_error[:500],
        )

        if exit_status != 0:
            error_msg = f"Command '{command}' failed with exit code {exit_status}."
            logger.error(error_msg)
            logger.error("Output:\n%s", output)
            logger.error("Error Output:\n%s", error_output)
            raise RuntimeError(error_msg)

        # For interactive commands, we don't return output, just ensure it ran successfully.
        # The original _execute_command would have logged it.

    def execute_command_capture(self, command: str, timeout: int = 30) -> str:
        """Execute a command and capture its output.

        Returns:
            The command output.

        Raises:
            RuntimeError: If the SSH session is not established or command fails.
        """
        if not self.client:
            self.connect()
        if not self.client:
            msg = "SSH session is not properly initialised."
            raise RuntimeError(msg)

        # Check if connection is still alive
        transport = self.client.get_transport()
        if not transport or not transport.is_active():
            logger.warning("SSH connection lost, reconnecting...")
            self.close()
            self.connect()

        logger.debug("Capturing output for command: %s", command)
        _stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        output = stdout.read().decode("utf-8")
        error_output = stderr.read().decode("utf-8")
        exit_status = stdout.channel.recv_exit_status()

        # Debug log sanitised SSH response
        sanitised_output = self._clean_output(output)
        sanitised_error = self._clean_output(error_output)
        logger.debug(
            "SSH Response [exit=%d]: stdout=%r stderr=%r",
            exit_status,
            sanitised_output[:500],
            sanitised_error[:500],
        )

        if exit_status != 0:
            error_msg = f"Command '{command}' failed with exit code {exit_status}."
            logger.error(error_msg)
            logger.error("Output:\n%s", output)
            logger.error("Error Output:\n%s", error_output)
            raise RuntimeError(error_msg)

        return output

    def execute_command_background(self, command: str) -> None:
        """Execute a command in the background without waiting for output.

        This method is designed for fire-and-forget commands that run in the
        background (e.g., using nohup, &, or disown). It returns immediately
        after launching the command without waiting for completion or output.

        Raises:
            RuntimeError: If the SSH session is not established.
        """
        if not self.client:
            self.connect()
        if not self.client:
            msg = "SSH session is not properly initialised."
            raise RuntimeError(msg)

        # Check if connection is still alive
        transport = self.client.get_transport()
        if not transport or not transport.is_active():
            logger.warning("SSH connection lost, reconnecting...")
            self.close()
            self.connect()

        logger.debug("Executing background command: %s", command)
        # Execute command without waiting for output
        self.client.exec_command(command)
        # Return immediately without reading stdout/stderr or waiting for exit status

    def _stream_output_chunk(self, stdout: ChannelFile, stderr: ChannelFile) -> None:
        """Process and log a chunk of streaming output."""
        if stdout.channel.recv_ready():
            chunk = stdout.read(1024).decode("utf-8", errors="replace")
            if chunk:
                # Log each chunk for real-time progress
                cleaned_chunk = self._clean_output(chunk)
                logger.info(cleaned_chunk.rstrip())
        if stderr.channel.recv_ready():
            chunk = stderr.read(1024).decode("utf-8", errors="replace")
            if chunk:
                cleaned_chunk = self._clean_output(chunk)
                logger.warning(cleaned_chunk.rstrip())

    def execute_command_streaming(self, command: str, timeout: int = 600) -> None:
        """Execute a command with real-time output streaming.

        This method provides real-time output streaming for long-running commands
        like model downloads, showing progress as it happens.

        Raises:
            RuntimeError: If the SSH session is not established or command fails.
        """
        if not self.client:
            self.connect()
        if not self.client:
            msg = "SSH session is not properly initialised."
            raise RuntimeError(msg)

        # Check if connection is still alive
        transport = self.client.get_transport()
        if not transport or not transport.is_active():
            logger.warning("SSH connection lost, reconnecting...")
            self.close()
            self.connect()

        logger.debug("Executing streaming command: %s", command)
        _stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)

        # Stream output in real-time
        while not stdout.channel.exit_status_ready():
            self._stream_output_chunk(stdout, stderr)
            # Small delay to prevent excessive polling
            time.sleep(0.1)

        # Get final exit status
        exit_status = stdout.channel.recv_exit_status()

        # Get any remaining output
        final_stdout = stdout.read().decode("utf-8", errors="replace")
        final_stderr = stderr.read().decode("utf-8", errors="replace")

        if final_stdout:
            logger.info(self._clean_output(final_stdout).rstrip())
        if final_stderr:
            logger.warning(self._clean_output(final_stderr).rstrip())

        if exit_status != 0:
            error_msg = f"Command '{command}' failed with exit code {exit_status}."
            raise RuntimeError(error_msg)

    def close(self) -> None:
        """Close the SSH and SFTP sessions."""
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.client:
            self.client.close()
            self.client = None

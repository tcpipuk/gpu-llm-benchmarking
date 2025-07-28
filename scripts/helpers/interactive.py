"""Interactive SSH session management for remote command execution.

This module provides functionality for executing commands on remote instances
using pexpect for real-time output streaming and interactive session management.
"""

from __future__ import annotations

import logging
import select
import time

import paramiko

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

    def _ensure_connected(self) -> None:
        """Ensure SSH client is connected and reconnect if necessary.

        Raises:
            RuntimeError: If the SSH session is not properly initialised.
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

    def _log_debug_response(self, output: str, error_output: str, exit_status: int) -> None:
        """Log debug information about command response."""
        sanitised_output = self._clean_output(output)
        sanitised_error = self._clean_output(error_output)
        logger.debug(
            "SSH Response [exit=%d]: stdout=%r stderr=%r",
            exit_status,
            sanitised_output[:500],
            sanitised_error[:500],
        )

    def _log_command_output(self, output: str, error_output: str) -> None:
        """Log the output from a command execution."""
        if output.strip():
            for line in output.strip().split("\n"):
                if line.strip():
                    logger.info(line.strip())

        if error_output.strip():
            for line in error_output.strip().split("\n"):
                if line.strip():
                    logger.warning(line.strip())

    def _check_command_success(self, command: str, exit_status: int) -> None:
        """Check if command succeeded and raise error if not.

        Raises:
            RuntimeError: If the command fails.
        """
        if exit_status != 0:
            error_msg = f"Command '{command}' failed with exit code {exit_status}."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _check_capture_command_success(
        self, command: str, exit_status: int, output: str, error_output: str
    ) -> None:
        """Check if capture command succeeded and raise error if not.

        Raises:
            RuntimeError: If the command fails.
        """
        if exit_status != 0:
            error_msg = f"Command '{command}' failed with exit code {exit_status}."
            logger.error(error_msg)
            logger.error("Output:\n%s", output)
            logger.error("Error Output:\n%s", error_output)
            raise RuntimeError(error_msg)

    def execute_command_interactive(self, command: str, timeout: int = 600) -> None:
        """Execute a command with real-time output.

        Raises:
            RuntimeError: If the SSH session is not established or command fails.
        """
        self._ensure_connected()
        if not self.client:
            msg = "SSH session is not properly initialised after connection attempt."
            raise RuntimeError(msg)

        logger.debug("Executing command: %s", command)
        _stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        output = stdout.read().decode("utf-8")
        error_output = stderr.read().decode("utf-8")
        exit_status = stdout.channel.recv_exit_status()

        self._log_debug_response(output, error_output, exit_status)
        self._log_command_output(output, error_output)
        self._check_command_success(command, exit_status)

    def execute_command_capture(self, command: str, timeout: int = 30) -> str:
        """Execute a command and capture its output.

        Returns:
            The command output.

        Raises:
            RuntimeError: If the SSH session is not established or command fails.
        """
        self._ensure_connected()
        if not self.client:
            msg = "SSH session is not properly initialised after connection attempt."
            raise RuntimeError(msg)

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
        self._ensure_connected()
        if not self.client:
            msg = "SSH session is not properly initialised after connection attempt."
            raise RuntimeError(msg)

        logger.debug("Executing background command: %s", command)
        # Execute command without waiting for output
        self.client.exec_command(command)
        # Add a small delay to allow the SSH client to process any immediate protocol messages
        time.sleep(0.1)
        # Return immediately without reading stdout/stderr or waiting for exit status

    def _log_output_line(self, line: str, is_stderr: bool) -> None:
        """Log a single output line appropriately."""
        if is_stderr:
            logger.warning(line.rstrip())
        else:
            logger.info(line.rstrip())

    def _process_chunk_data(
        self, chunk: str, line_buffer: str, last_line: str, is_stderr: bool
    ) -> tuple[str, str]:
        """Process a chunk of data and return updated buffer and last line.

        Returns:
            A tuple containing the updated line_buffer and last_line.
        """
        for char in chunk:
            if char == "\r":
                # Carriage return: clear buffer, next output will overwrite current line
                if line_buffer.strip() and line_buffer.strip() != last_line:
                    self._log_output_line(line_buffer, is_stderr)
                    last_line = line_buffer.strip()
                line_buffer = ""
            elif char == "\n":
                # Newline: process the accumulated line
                if line_buffer.strip() and line_buffer.strip() != last_line:
                    self._log_output_line(line_buffer, is_stderr)
                    last_line = line_buffer.strip()
                line_buffer = ""
            else:
                line_buffer += char
        return line_buffer, last_line

    def _process_stream_output(
        self,
        channel: paramiko.Channel,
        is_stderr: bool,
        last_stdout_line: str,
        last_stderr_line: str,
        line_buffer: str,
    ) -> tuple[str, str, str]:
        """Helper to process output from a channel, handling carriage returns and duplicates.

        Returns:
            A tuple containing the updated last_stdout_line, last_stderr_line, and line_buffer.
        """
        if is_stderr:
            chunk = channel.recv_stderr(1024).decode("utf-8", errors="replace")
            last_line = last_stderr_line
        else:
            chunk = channel.recv(1024).decode("utf-8", errors="replace")
            last_line = last_stdout_line

        if chunk:
            line_buffer, last_line = self._process_chunk_data(
                chunk, line_buffer, last_line, is_stderr
            )

        if is_stderr:
            return last_stdout_line, last_line, line_buffer
        return last_line, last_stderr_line, line_buffer

    def _process_channel_streams(self, channel: paramiko.Channel) -> tuple[str, str, str, str]:
        """Process stdout and stderr streams from channel.

        Returns:
            Tuple of (last_stdout_line, last_stderr_line, stdout_buffer, stderr_buffer)
        """
        last_stdout_line = ""
        last_stderr_line = ""
        stdout_buffer = ""
        stderr_buffer = ""

        while (
            not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready()
        ):
            rlist, _, _ = select.select([channel], [], [], 0.1)

            if rlist:
                if channel.recv_ready():
                    last_stdout_line, last_stderr_line, stdout_buffer = self._process_stream_output(
                        channel, False, last_stdout_line, last_stderr_line, stdout_buffer
                    )
                if channel.recv_stderr_ready():
                    last_stdout_line, last_stderr_line, stderr_buffer = self._process_stream_output(
                        channel, True, last_stdout_line, last_stderr_line, stderr_buffer
                    )
            elif not channel.exit_status_ready():
                time.sleep(0.1)

        return last_stdout_line, last_stderr_line, stdout_buffer, stderr_buffer

    def _flush_remaining_output(
        self, channel: paramiko.Channel, buffers: tuple[str, str, str, str]
    ) -> None:
        """Flush any remaining output after command completion."""
        last_stdout_line, last_stderr_line, stdout_buffer, stderr_buffer = buffers

        # Read remaining output
        while channel.recv_ready() or channel.recv_stderr_ready():
            if channel.recv_ready():
                last_stdout_line, last_stderr_line, stdout_buffer = self._process_stream_output(
                    channel, False, last_stdout_line, last_stderr_line, stdout_buffer
                )
            if channel.recv_stderr_ready():
                last_stdout_line, last_stderr_line, stderr_buffer = self._process_stream_output(
                    channel, True, last_stdout_line, last_stderr_line, stderr_buffer
                )

        # Process any remaining buffer content
        if stdout_buffer.strip() and stdout_buffer.strip() != last_stdout_line:
            logger.info(stdout_buffer.rstrip())
        if stderr_buffer.strip() and stderr_buffer.strip() != last_stderr_line:
            logger.warning(stderr_buffer.rstrip())

    def execute_command_streaming(self, command: str, timeout: int = 600) -> None:
        """Execute a command with real-time output streaming.

        This method provides real-time output streaming for long-running commands
        like model downloads, showing progress as it happens.

        Raises:
            RuntimeError: If the SSH session is not established or command fails.
        """
        self._ensure_connected()
        if not self.client:
            msg = "SSH session is not properly initialised after connection attempt."
            raise RuntimeError(msg)

        logger.debug("Executing streaming command: %s", command)
        _stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)

        # Make channels non-blocking
        stdout.channel.setblocking(0)
        stderr.channel.setblocking(0)

        channel = stdout.channel

        # Process streams and get final buffers
        buffers = self._process_channel_streams(channel)

        # Flush any remaining output
        self._flush_remaining_output(channel, buffers)

        exit_status = channel.recv_exit_status()
        self._check_command_success(command, exit_status)

    def close(self) -> None:
        """Close the SSH and SFTP sessions."""
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.client:
            self.client.close()
            self.client = None

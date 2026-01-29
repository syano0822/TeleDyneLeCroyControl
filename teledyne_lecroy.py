#!/usr/bin/env python3
"""
Teledyne LeCroy oscilloscope library.

Supports WavePro and WaveRunner series oscilloscopes via VISA/TCP.

Bong-Hwi Lim (UTokyo)
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import numpy as np
import pyvisa
from numpy.typing import NDArray
from pyvisa.resources import MessageBasedResource


__all__ = [
    # Enums
    "TriggerSlope",
    "TriggerState",
    "Coupling",
    "AuxOutputMode",
    # Exceptions
    "ScopeError",
    "ScopeConnectionError",
    "ScopeTimeoutError",
    "ScopeConfigurationError",
    "ScopeTriggerError",
    # Config dataclasses
    "ChannelConfig",
    "ChannelTrigger",
    "TriggerConfig",
    "AcquisitionConfig",
    "SequenceConfig",
    # Data dataclasses
    "WaveformData",
    "SequenceData",
    # Classes
    "TeledyneLecroyScope",
    "WavePro",
    "WaveRunner",
]


class TriggerSlope(Enum):
    """Trigger edge direction (legacy, use TriggerState for pattern trigger)."""
    RISING = auto()
    FALLING = auto()
    EITHER = auto()


class TriggerState(Enum):
    """Per-channel trigger state for pattern trigger."""
    HIGH = "H"       # Trigger on high level (rising edge)
    LOW = "L"        # Trigger on low level (falling edge)
    DONT_CARE = "X"  # Don't use this channel for triggering


class Coupling(Enum):
    """Channel input coupling/termination."""
    DC50 = "D50"   # 50Ω DC
    DC1M = "D1M"   # 1MΩ DC
    AC1M = "A1M"   # 1MΩ AC


class AuxOutputMode(Enum):
    """Auxiliary output mode."""
    TRIGGER_OUT = "TriggerOut"           # Output trigger pulse
    TRIGGER_ENABLED = "TriggerEnabled"   # Output when trigger is armed/enabled


# === Exceptions ===

class ScopeError(Exception):
    """Base exception for Teledyne LeCroy scope errors."""
    pass


class ScopeConnectionError(ScopeError):
    """Connection failure (network, VISA, etc.)."""
    pass


class ScopeTimeoutError(ScopeError):
    """Response timeout."""
    pass


class ScopeConfigurationError(ScopeError):
    """Invalid configuration value."""
    pass


class ScopeTriggerError(ScopeError):
    """Trigger-related error (no signal, etc.)."""
    pass


# === Configuration Dataclasses ===

@dataclass
class ChannelConfig:
    """Single channel configuration."""
    vdiv: float = 0.020            # V/div
    offset: float = 0.0            # V
    coupling: Coupling = Coupling.DC50
    enabled: bool = True


@dataclass
class ChannelTrigger:
    """Per-channel trigger configuration.

    Each channel can be HIGH, LOW, or DONT_CARE (X).
    Level can be absolute or relative to baseline.
    """
    state: TriggerState = TriggerState.DONT_CARE
    level: float | None = None     # V absolute (if set, ignores level_offset)
    level_offset: float = 0.0      # V relative to baseline (used if level is None)


@dataclass
class TriggerConfig:
    """Trigger configuration with per-channel settings.

    Example:
        TriggerConfig(
            channels={
                1: ChannelTrigger(state=TriggerState.HIGH, level=0.1),
                2: ChannelTrigger(state=TriggerState.LOW, level=-0.05),
            },
            mode="SINGLE"
        )

    External trigger example:
        TriggerConfig(
            external=True,
            external_level=1.25,  # V
            channels={1: ChannelTrigger(state=TriggerState.HIGH, level=0.1)},
        )
    """
    channels: dict[int, ChannelTrigger] = field(default_factory=dict)
    mode: Literal["AUTO", "NORM", "SINGLE", "STOP"] = "SINGLE"
    external: bool = False         # Use external trigger (EX)
    external_level: float = 1.25   # External trigger level (V)


@dataclass
class AcquisitionConfig:
    """Acquisition timing configuration."""
    tdiv: float = 5e-9             # s/div
    sampling_period: float = 25e-12  # s (25ps = 40GS/s)
    trigger_delay: float = 0.0     # s
    window_delay: float = 10e-9    # s


@dataclass
class SequenceConfig:
    """Sequence mode configuration."""
    enabled: bool = False
    num_segments: int = 1
    timeout_enabled: bool = False
    timeout_seconds: float = 2.5e6


# === Data Dataclasses ===

@dataclass(frozen=True)
class WaveformData:
    """Single segment waveform data (immutable)."""
    raw_data: bytes
    channel: int
    segment: int = 0               # Segment index (0-based)
    dx: float = 0.0                # Time step (s)
    x0: float = 0.0                # Time origin (s)
    dy: float = 0.0                # Voltage scale (V/ADC)
    y0: float = 0.0                # Voltage offset (V)
    trigger_time: str | None = None

    def to_voltage(self) -> NDArray[np.float64]:
        """Convert raw bytes to voltage array."""
        raw = np.frombuffer(self.raw_data, dtype=np.int8)
        return raw * self.dy + self.y0

    def to_time(self) -> NDArray[np.float64]:
        """Generate time axis array."""
        n_points = len(self.raw_data)
        return np.arange(n_points) * self.dx + self.x0


@dataclass(frozen=True)
class SequenceData:
    """Sequence mode data container (immutable)."""
    segments: tuple[WaveformData, ...]
    channel: int

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> WaveformData:
        return self.segments[idx]

    def to_voltage_array(self) -> NDArray[np.float64]:
        """Convert all segments to 2D array (n_segments, n_points)."""
        return np.array([seg.to_voltage() for seg in self.segments])


# === Base Class ===

class TeledyneLecroyScope(ABC):
    """Abstract base class for Teledyne LeCroy oscilloscopes."""

    # Hardware specs (override in subclasses)
    MAX_SAMPLING_RATE: float = 40e9   # samples/s
    MIN_MEMORY_SIZE: int = 500        # points
    TIME_DIVISIONS: int = 10
    VOLTAGE_DIVISIONS: int = 8
    MAX_CHANNELS: int = 4
    DY_ADC_CONVERSION: float = 0.03125  # V/ADC = vdiv * 8 / 256
    X0_DIVISION: int = -5  # Time divisions for x0 calculation

    def __init__(
        self,
        address: str,
        timeout: float = 30.0,
        active_channels: list[int] | None = None,
    ) -> None:
        self._address = address
        self._timeout = timeout
        self._active_channels = active_channels or [1, 2, 3, 4]
        self._scope: MessageBasedResource | None = None
        self._rm: pyvisa.ResourceManager | None = None
        self._connected = False
        self._logger = logging.getLogger(self.__class__.__name__)

        # State tracking
        self._channel_configs: dict[int, ChannelConfig] = {}
        self._acquisition_config: AcquisitionConfig | None = None
        self._trigger_config: TriggerConfig | None = None
        self._sequence_config: SequenceConfig | None = None

        # Validate channels
        for ch in self._active_channels:
            if not 1 <= ch <= self.MAX_CHANNELS:
                raise ScopeConfigurationError(f"Invalid channel: {ch}")

    def __enter__(self) -> TeledyneLecroyScope:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def connect(self) -> None:
        """Establish connection to scope."""
        if self._connected:
            return

        try:
            self._rm = pyvisa.ResourceManager()
            self._scope = self._rm.open_resource(
                f"TCPIP0::{self._address}::inst0::INSTR",
                resource_pyclass=MessageBasedResource,
            )
            self._scope.timeout = int(self._timeout * 1000)
            self._connected = True

            # Suppress pyvisa logging
            logging.getLogger("pyvisa").setLevel(logging.WARNING)

            idn = self.query("*IDN?")
            self._logger.info(f"Connected: {idn}")

        except pyvisa.Error as e:
            raise ScopeConnectionError(f"Connection failed: {self._address}") from e

    def disconnect(self) -> None:
        """Close connection to scope."""
        if self._scope:
            self._scope.close()
        if self._rm:
            self._rm.close()
        self._scope = None
        self._rm = None
        self._connected = False
        self._logger.info("Disconnected")

    # === Low-level Communication ===

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected or self._scope is None:
            raise ScopeConnectionError("Not connected to scope")

    def write(self, command: str) -> None:
        """Send SCPI command."""
        self._ensure_connected()
        self._scope.write(command)
        self._logger.debug(f"WRITE: {command}")

    def query(self, command: str) -> str:
        """Send SCPI query and return response."""
        self._ensure_connected()
        response = self._scope.query(command).strip()
        self._logger.debug(f"QUERY: {command} -> {response}")
        return response

    def read_raw(self) -> bytes:
        """Read raw binary data."""
        self._ensure_connected()
        return self._scope.read_raw()

    def wait_opc(self, timeout: float | None = None) -> None:
        """Wait for operation complete (*OPC?)."""
        self._ensure_connected()
        old_timeout = self._scope.timeout
        if timeout:
            self._scope.timeout = int(timeout * 1000)
        try:
            while True:
                opc = self.query("*OPC?").strip()
                if opc == "1":
                    break
                time.sleep(0.01)
        finally:
            self._scope.timeout = old_timeout

    def clear(self) -> None:
        """Clear scope registers and buffers."""
        self._ensure_connected()
        self._scope.clear()
        self._logger.info("Scope cleared")

    # === Configuration ===

    def configure(
        self,
        channels: dict[int, ChannelConfig],
        acquisition: AcquisitionConfig,
        sequence: SequenceConfig | None = None,
        display: bool = False,
    ) -> None:
        """Configure scope with given settings.

        Args:
            channels: Channel configurations
            acquisition: Acquisition/timebase configuration
            sequence: Optional sequence mode configuration
            display: Keep display on (True) or off for faster acquisition (False)
        """
        self._validate_config(channels, acquisition)

        self.write("CHDR OFF")  # Remove response headers
        self._configure_display(display=display)
        self._configure_timebase(acquisition)

        for ch, cfg in channels.items():
            if cfg.enabled:
                self._configure_channel(ch, cfg)

        if sequence and sequence.enabled:
            self._configure_sequence(sequence)

        self._channel_configs = channels
        self._acquisition_config = acquisition
        self._sequence_config = sequence
        self._logger.info("Configuration complete")

    def _validate_config(
        self,
        channels: dict[int, ChannelConfig],
        acquisition: AcquisitionConfig,
    ) -> None:
        """Validate configuration values."""
        for ch in channels:
            if not 1 <= ch <= self.MAX_CHANNELS:
                raise ScopeConfigurationError(f"Invalid channel: {ch}")

        max_window = self.TIME_DIVISIONS / 2 * acquisition.tdiv
        if acquisition.window_delay > max_window:
            raise ScopeConfigurationError(
                f"window_delay ({acquisition.window_delay}s) exceeds max ({max_window}s)"
            )

    def _calculate_memory_size(self, acquisition: AcquisitionConfig) -> float:
        """Calculate memory size based on sampling rate."""
        return (
            self.TIME_DIVISIONS * acquisition.tdiv / acquisition.sampling_period
        )

    def _calculate_num_points(self, acquisition: AcquisitionConfig) -> int:
        """Calculate number of acquisition points."""
        return int(
            self.TIME_DIVISIONS * acquisition.tdiv / acquisition.sampling_period
        )

    # === Read Settings from Scope ===

    def _parse_numeric_response(self, response: str) -> float:
        """Parse numeric value from SCPI response.

        Handles formats like:
        - "5E-6" (value only)
        - "TDIV 5E-6 S" (command + value + unit)
        - "C1:VDIV 2E-1 V" (channel:command + value + unit)
        """
        # Try to find a numeric value in the response
        for part in response.split():
            # Skip parts that look like commands or units
            if ":" in part or part.isalpha():
                continue
            try:
                return float(part)
            except ValueError:
                continue

        # If no numeric found, try the whole string (minus trailing unit)
        parts = response.split()
        if len(parts) >= 2:
            try:
                return float(parts[-2])  # Value is typically second-to-last
            except ValueError:
                pass

        raise ValueError(f"Could not parse numeric value from: {response}")

    def read_channel_config(self, channel: int) -> ChannelConfig:
        """Read current channel configuration from scope."""
        self._ensure_connected()
        if not 1 <= channel <= self.MAX_CHANNELS:
            raise ScopeConfigurationError(f"Invalid channel: {channel}")

        vdiv = self._parse_numeric_response(self.query(f"C{channel}:VDIV?"))
        offset = self._parse_numeric_response(self.query(f"C{channel}:OFST?"))
        trace_state = self.query(f"C{channel}:TRA?")
        enabled = "ON" in trace_state.upper()

        return ChannelConfig(vdiv=vdiv, offset=offset, enabled=enabled)

    def read_acquisition_config(self) -> AcquisitionConfig:
        """Read current acquisition configuration from scope."""
        self._ensure_connected()

        tdiv = self._parse_numeric_response(self.query("TDIV?"))

        # Calculate sampling period from current memory size
        memory_response = self.query("MSIZ?")
        # Parse memory size (e.g., "10000", "10K", "1M", or "MSIZ 10000")
        memory_str = memory_response.upper()
        # Remove command prefix if present
        if "MSIZ" in memory_str:
            memory_str = memory_str.replace("MSIZ", "").strip()
        # Handle K/M suffixes
        memory_str = memory_str.replace("K", "E3").replace("M", "E6").replace("SA", "").strip()
        try:
            memory_size = self._parse_numeric_response(memory_str)
            sampling_period = self.TIME_DIVISIONS * tdiv / memory_size
        except ValueError:
            # Fallback to a reasonable default
            self._logger.warning(f"Could not parse memory size: {memory_response}")
            sampling_period = tdiv / 1000

        return AcquisitionConfig(tdiv=tdiv, sampling_period=sampling_period)

    def read_trigger_level(self, channel: int) -> float:
        """Read current trigger level for a channel."""
        self._ensure_connected()
        return self._parse_numeric_response(self.query(f"C{channel}:TRLV?"))

    def read_trigger_pattern(self) -> dict[int, str]:
        """Read current trigger pattern states for all channels.

        Returns:
            Dict mapping channel number to state ("H", "L", or "X")
        """
        self._ensure_connected()
        response = self.query("TRPA?")
        self._logger.debug(f"Trigger pattern response: {response}")

        # Parse response like "TRPA C1,H,C2,L,C3,X,C4,X,STATE,OR"
        states = {}
        parts = response.replace("TRPA", "").strip().split(",")
        for i in range(0, len(parts) - 2, 2):  # Skip last "STATE,OR" part
            ch_part = parts[i].strip()
            if ch_part.startswith("C") and len(parts) > i + 1:
                try:
                    ch = int(ch_part[1:])
                    state = parts[i + 1].strip().upper()
                    if state in ("H", "L", "X"):
                        states[ch] = state
                except (ValueError, IndexError):
                    pass

        return states

    def read_all_settings(self) -> dict:
        """Read all current settings from scope as a dictionary."""
        self._ensure_connected()

        # Read ALL channel settings (not just enabled)
        channels = {}
        for ch in range(1, self.MAX_CHANNELS + 1):
            try:
                config = self.read_channel_config(ch)
                channels[str(ch)] = {
                    "vdiv": config.vdiv,
                    "offset": config.offset,
                    "enabled": config.enabled,
                }
            except Exception as e:
                self._logger.debug(f"Could not read channel {ch}: {e}")

        # Read acquisition settings
        acq = self.read_acquisition_config()
        acquisition = {
            "tdiv": acq.tdiv,
            "sampling_period": acq.sampling_period,
        }

        # Read trigger mode
        trigger_response = self.query("TRMD?").strip()
        trigger_mode = trigger_response.split()[-1] if trigger_response else "SINGLE"

        # Read trigger pattern states
        try:
            pattern_states = self.read_trigger_pattern()
        except Exception as e:
            self._logger.debug(f"Could not read trigger pattern: {e}")
            pattern_states = {}

        # Build per-channel trigger config
        trigger_channels = {}
        state_map = {"H": "HIGH", "L": "LOW", "X": "DONT_CARE"}

        for ch in range(1, self.MAX_CHANNELS + 1):
            try:
                level = self.read_trigger_level(ch)
                state_code = pattern_states.get(ch, "X")
                state_name = state_map.get(state_code, "DONT_CARE")

                trigger_channels[str(ch)] = {
                    "state": state_name,
                    "level": level,
                }
            except Exception as e:
                self._logger.debug(f"Could not read trigger for channel {ch}: {e}")

        trigger = {
            "channels": trigger_channels,
            "mode": trigger_mode,
        }

        return {
            "channels": channels,
            "acquisition": acquisition,
            "trigger": trigger,
        }

    def save_settings(self, filepath: str | Path) -> None:
        """Save current scope settings to JSON file."""
        settings = self.read_all_settings()
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(settings, f, indent=2)
        self._logger.info(f"Settings saved to {filepath}")

    @staticmethod
    def load_settings_file(filepath: str | Path) -> dict:
        """Load settings from JSON file (static method, no scope needed)."""
        filepath = Path(filepath)
        with filepath.open() as f:
            return json.load(f)

    # === Offset and Auxiliary Output ===

    def set_offset(
        self,
        channels: dict[int, float] | None = None,
        search: bool = False,
    ) -> dict[int, float]:
        """Set channel offsets.

        Args:
            channels: Dict of channel -> offset (V). If None, uses all active channels.
            search: If True, automatically find signal and set offset.

        Returns:
            Dict of channel -> actual offset set (V).
        """
        if search:
            return self._auto_offset_search()

        if channels is None:
            channels = {ch: 0.0 for ch in self._active_channels}

        for ch, offset in channels.items():
            self.write(f"C{ch}:OFFSET {-offset}")
            self._logger.debug(f"Channel {ch} offset: {offset}V")

        return channels

    def set_auxiliary_output(self, mode: AuxOutputMode) -> None:
        """Set auxiliary output mode.

        Args:
            mode: AuxOutputMode.TRIGGER_OUT or AuxOutputMode.TRIGGER_ENABLED
        """
        self.write(f"""vbs 'app.Acquisition.AuxOutput.Mode = "{mode.value}"' """)
        self._logger.info(f"Auxiliary output: {mode.value}")

    @abstractmethod
    def _auto_offset_search(self) -> dict[int, float]:
        """Automatically find signal and set offset for each channel."""
        ...

    # === Abstract Methods (implement in subclasses) ===

    @abstractmethod
    def _configure_display(self, display: bool = False) -> None:
        """Configure display settings.

        Args:
            display: Keep display on (True) or off for faster acquisition (False)
        """
        ...

    @abstractmethod
    def _configure_timebase(self, config: AcquisitionConfig) -> None:
        """Configure timebase settings."""
        ...

    @abstractmethod
    def _configure_channel(self, channel: int, config: ChannelConfig) -> None:
        """Configure single channel."""
        ...

    @abstractmethod
    def _configure_sequence(self, config: SequenceConfig) -> None:
        """Configure sequence mode."""
        ...

    # === Trigger ===

    def set_trigger(self, config: TriggerConfig) -> None:
        """Configure trigger settings."""
        self._trigger_config = config
        self._setup_trigger_source(config)
        self._setup_trigger_level(config)

        # Log trigger configuration
        active = [f"CH{ch}:{t.state.name}" for ch, t in config.channels.items()
                  if t.state != TriggerState.DONT_CARE]
        self._logger.info(f"Trigger configured: {', '.join(active) or 'none'}")

    def arm(self, force: bool = False) -> None:
        """Arm trigger and wait for acquisition."""
        if force:
            self.write("TRMD NORM")
            self.write("FRTR")  # Force trigger
            self._logger.debug("Forced trigger")
        else:
            mode = self._trigger_config.mode if self._trigger_config else "SINGLE"
            self.write(f"TRMD {mode}")
            self.wait_opc()
            self._logger.debug(f"Armed with mode {mode}, waiting for trigger")

    def is_triggered(self) -> bool:
        """Check if trigger has occurred."""
        return "STOP" in self.query("TRMD?")

    def wait_for_trigger(self, timeout: float | None = None, force: bool = False) -> None:
        """Wait for trigger with optional timeout.

        Args:
            timeout: Maximum time to wait in seconds
            force: If True, use force trigger (useful for testing without signal)
        """
        start = time.time()

        # For sequence mode with force trigger - fill all segments immediately
        if force and self._sequence_config and self._sequence_config.enabled:
            num_segments = self._sequence_config.num_segments
            for i in range(num_segments):
                self.write("FRTR")
                time.sleep(0.01)  # Small delay between force triggers
                if timeout and (time.time() - start) > timeout:
                    raise ScopeTimeoutError(f"Trigger timeout: {timeout}s")
            self.write("TRMD STOP")
            self.wait_opc()
            self._logger.debug(f"Forced {num_segments} triggers for sequence capture")
            return

        # For sequence mode, use WAIT command which blocks until all segments captured
        if self._sequence_config and self._sequence_config.enabled:
            old_timeout = self._scope.timeout
            if timeout:
                self._scope.timeout = int(timeout * 1000)  # Convert to ms
            try:
                self.write("WAIT")
                self.wait_opc()
                self._logger.debug("Sequence acquisition complete")
            except Exception as e:
                raise ScopeTimeoutError(f"Trigger timeout: {timeout}s") from e
            finally:
                self._scope.timeout = old_timeout
            return

        # For normal mode, poll TRMD status
        while not self.is_triggered():
            if timeout and (time.time() - start) > timeout:
                raise ScopeTimeoutError(f"Trigger timeout: {timeout}s")
            time.sleep(0.01)

    def set_trigger_mode(self, mode: Literal["AUTO", "NORM", "SINGLE", "STOP"]) -> None:
        """Set trigger sweep mode."""
        self.write(f"TRMD {mode}")
        self._logger.debug(f"Trigger mode: {mode}")

    @abstractmethod
    def _setup_trigger_source(self, config: TriggerConfig) -> None:
        """Setup trigger source and slope."""
        ...

    @abstractmethod
    def _setup_trigger_level(self, config: TriggerConfig) -> None:
        """Setup trigger levels for each channel."""
        ...

    @abstractmethod
    def _measure_baseline(self, channel: int) -> float:
        """Measure baseline voltage for trigger level calculation."""
        ...

    # === Readout ===

    def readout(self, channels: list[int] | None = None) -> dict[int, WaveformData]:
        """Read waveform data from specified channels."""
        channels = channels or list(self._channel_configs.keys())
        result: dict[int, WaveformData] = {}

        trigger_time = self._get_trigger_time()

        for ch in channels:
            raw = self._read_channel_data(ch)
            dx, x0, dy, y0 = self._get_waveform_scaling(ch)

            result[ch] = WaveformData(
                raw_data=raw,
                channel=ch,
                segment=0,
                dx=dx,
                x0=x0,
                dy=dy,
                y0=y0,
                trigger_time=trigger_time,
            )

        self._logger.debug(f"Readout complete: channels {channels}")
        return result

    def readout_sequence(
        self, channels: list[int] | None = None
    ) -> dict[int, SequenceData]:
        """Read sequence mode data from specified channels."""
        channels = channels or list(self._channel_configs.keys())
        result: dict[int, SequenceData] = {}

        if not self._sequence_config or not self._sequence_config.enabled:
            raise ScopeConfigurationError("Sequence mode not enabled")

        for ch in channels:
            segments = self._read_sequence_segments(ch)
            result[ch] = SequenceData(segments=tuple(segments), channel=ch)

        self._logger.debug(f"Sequence readout complete: channels {channels}")
        return result

    @abstractmethod
    def _read_channel_data(self, channel: int) -> bytes:
        """Read raw waveform data from channel."""
        ...

    @abstractmethod
    def _get_waveform_scaling(
        self, channel: int
    ) -> tuple[float, float, float, float]:
        """Get scaling factors (dx, x0, dy, y0) for channel."""
        ...

    @abstractmethod
    def _get_trigger_time(self) -> str | None:
        """Get trigger timestamp."""
        ...

    @abstractmethod
    def _read_sequence_segments(self, channel: int) -> list[WaveformData]:
        """Read all sequence segments for channel."""
        ...


# === WavePro ===

class WavePro(TeledyneLecroyScope):
    """Teledyne LeCroy WavePro series oscilloscope."""

    MAX_SAMPLING_RATE: float = 80e9   # 80 GS/s
    MIN_MEMORY_SIZE: int = 500
    MAX_BANDWIDTH: float = 8e9        # 8 GHz
    ADC_BITS: int = 8

    def _configure_display(self, display: bool = False) -> None:
        """Configure display for remote operation."""
        if display:
            self.write("DISP ON")
        else:
            self.write("DISP OFF")
        self.write("CHDR OFF")
        self.write("GRID QUATTRO")

    def _configure_timebase(self, config: AcquisitionConfig) -> None:
        """Configure timebase and memory."""
        # Calculate memory size
        memory_size = self._calculate_memory_size(config)
        if memory_size < self.MIN_MEMORY_SIZE:
            memory_size = self.MIN_MEMORY_SIZE

        self.write(f"MSIZ {memory_size}")
        self.write(f"TDIV {config.tdiv}")
        self.write(f"TRDL {config.trigger_delay}")

        # Waveform setup for all channels
        num_points = self._calculate_num_points(config)
        sparsification = 1
        first_point = int(
            config.window_delay / config.sampling_period
        ) - (num_points // 2)

        for ch in self._active_channels:
            self.write(
                f"C{ch}:WFSU SP,{sparsification},NP,{num_points},"
                f"FP,{first_point},SN,0"
            )

        self._logger.info(f"Timebase: TDIV={config.tdiv}, points={num_points}")

    def _configure_channel(self, channel: int, config: ChannelConfig) -> None:
        """Configure single channel."""
        ch = f"C{channel}"
        self.write(f"{ch}:TRACE ON")
        self.write(f"{ch}:VDIV {config.vdiv}")
        self.write(f"{ch}:OFFSET {-config.offset}")
        self.write(f"{ch}:CPL {config.coupling.value}")
        self.write(f"{ch}:ATTN 1")
        self.write("BWL OFF")

        self._logger.debug(f"Channel {channel} configured: vdiv={config.vdiv}")

    def _configure_sequence(self, config: SequenceConfig) -> None:
        """Configure sequence mode."""
        self.write(f"SEQ ON,{config.num_segments},2.5E+6")
        self._logger.info(f"Sequence mode: {config.num_segments} segments")

        if config.timeout_enabled:
            self.write(
                fr"""vbs 'app.Acquisition.Horizontal.SequenceTimeoutEnable = -1' """
            )
            self.write(
                fr"""vbs 'app.Acquisition.Horizontal.SequenceTimeout = {config.timeout_seconds}' """
            )

    def _get_trigger_channels(self, config: TriggerConfig) -> dict[int, ChannelTrigger]:
        """Get channel trigger settings."""
        return config.channels

    def _setup_trigger_source(self, config: TriggerConfig) -> None:
        """Setup trigger source with pattern trigger."""
        channels = self._get_trigger_channels(config)

        states = ["X"] * 4
        for ch, ch_trig in channels.items():
            if 1 <= ch <= 4:
                states[ch - 1] = ch_trig.state.value

        # External trigger state
        ext_state = "H" if config.external else "X"

        self.write(
            f"TRPA C1,{states[0]},C2,{states[1]},C3,{states[2]},C4,{states[3]},"
            f"EX,{ext_state},STATE,OR"
        )
        self.write("TRIG_SELECT PA,SR,C1")

        if config.external:
            self._logger.debug(
                f"Trigger pattern: C1={states[0]}, C2={states[1]}, C3={states[2]}, C4={states[3]}, EX={ext_state}"
            )
        else:
            self._logger.debug(
                f"Trigger pattern: C1={states[0]}, C2={states[1]}, C3={states[2]}, C4={states[3]}"
            )

    def _setup_trigger_level(self, config: TriggerConfig) -> None:
        """Setup trigger level for each channel (absolute or relative to baseline)."""
        channels = self._get_trigger_channels(config)

        for ch, ch_trig in channels.items():
            if ch_trig.state == TriggerState.DONT_CARE:
                continue  # Skip channels not used for triggering

            if ch_trig.level is not None:
                # Use absolute level directly
                level = ch_trig.level
                self._logger.debug(f"Channel {ch} trigger level (absolute): {level}")
            else:
                # Measure baseline and add offset
                baseline = self._measure_baseline(ch)
                level = baseline + ch_trig.level_offset
                self._logger.debug(f"Channel {ch} trigger level (baseline={baseline}, offset={ch_trig.level_offset}): {level}")

            self.write(f"C{ch}:TRLV {level}")

        # External trigger level
        if config.external:
            self.write(f"EX:TRLV {config.external_level}")
            self._logger.debug(f"External trigger level: {config.external_level}V")

    # Constants for offset search
    _MIN_SIGNAL = 0.001      # V minimum amplitude to detect signal
    _V_DIVISION = 6          # Divisions to scan for signal
    _SHIFT_DIVISION = 2.5    # Divisions to shift after finding signal
    _MAX_OFFSET = -1.0       # V maximum offset before giving up

    def _auto_offset_search(self) -> dict[int, float]:
        """Automatically find signal and set offset for each channel."""
        offsets = {}

        for ch in self._active_channels:
            cfg = self._channel_configs.get(ch)
            vdiv = cfg.vdiv if cfg else 0.020

            # Setup measurements for this channel
            self.write(f"PACU 1,MEAN,C{ch}")  # Baseline
            self.write(f"PACU 2,AMPL,C{ch}")  # Amplitude

            # Start with zero offset
            initial_offset = 0.0
            self.write(f"C{ch}:OFFSET 0")

            # Scan for signal
            self.write("TRMD AUTO")
            time.sleep(1.0)
            self.write("TRMD STOP")

            amplitude_response = self.query(
                r"""vbs? 'return=app.measure.p2.out.result.value' """
            )

            # Try to find signal by scanning offsets
            iteration = 0
            while True:
                try:
                    amplitude = float(amplitude_response.split()[-1])
                    if amplitude >= self._MIN_SIGNAL:
                        break
                except (ValueError, IndexError):
                    pass

                # Move offset to search for signal
                initial_offset -= vdiv * self._V_DIVISION
                if initial_offset < self._MAX_OFFSET:
                    self._logger.warning(
                        f"Channel {ch}: Could not find signal, using offset=0"
                    )
                    offsets[ch] = 0.0
                    break

                self.write(f"C{ch}:OFFSET {initial_offset}")
                self.write("TRMD AUTO")
                time.sleep(1.0)
                self.write("TRMD NORM")

                amplitude_response = self.query(
                    r"""vbs? 'return=app.measure.p2.out.result.value' """
                )

                iteration += 1
                if iteration > 10:
                    self._logger.warning(
                        f"Channel {ch}: Max iterations, using offset=0"
                    )
                    offsets[ch] = 0.0
                    break
            else:
                continue

            # Found signal - measure baseline and set offset
            baseline_response = self.query(
                r"""vbs? 'return=app.measure.p1.out.result.value' """
            )
            try:
                baseline = float(baseline_response.split()[-1])
            except (ValueError, IndexError):
                baseline = 0.0

            # Shift signal to visible area
            offset = vdiv * self._SHIFT_DIVISION - baseline
            self.write(f"C{ch}:OFFSET {offset}")
            offsets[ch] = -offset  # Return positive offset value

            self._logger.info(f"Channel {ch}: auto offset = {-offset:.4f}V")

        self.write("TRMD NORM")
        return offsets

    def _measure_baseline(self, channel: int) -> float:
        """Measure baseline using parameter measurement."""
        self.write(f"PACU 1,MEAN,C{channel}")

        # Trigger briefly to get measurement
        self.write("TRMD AUTO")
        time.sleep(0.5)
        self.write("TRMD NORM")

        response = self.query(
            r"""vbs? 'return=app.measure.p1.out.result.value' """
        )
        self._logger.debug(f"Baseline measurement response: {response}")

        # Extract numeric value from response
        # Response format can vary; try to find a valid float
        for part in response.split():
            try:
                return float(part)
            except ValueError:
                continue

        # If no numeric value found, use 0 as baseline (DC level)
        self._logger.warning(
            f"Could not parse baseline for C{channel}, using 0. "
            f"Response was: {response}"
        )
        return 0.0

    def _read_channel_data(self, channel: int) -> bytes:
        """Read raw waveform data."""
        self.write(f"C{channel}:WF? DAT1")
        return self.read_raw()

    def _get_waveform_scaling(
        self, channel: int
    ) -> tuple[float, float, float, float]:
        """Get waveform scaling factors.

        If configure() was not called, reads values directly from scope.
        """
        # Get sampling period (dx)
        if self._acquisition_config is not None:
            dx = self._acquisition_config.sampling_period
        else:
            # Read from scope: calculate from TDIV and memory size
            acq = self.read_acquisition_config()
            dx = acq.sampling_period

        # Get time origin (x0) - always read from scope for accuracy
        tdiv = self._parse_numeric_response(self.query("TDIV?"))
        trdl = self._parse_numeric_response(self.query("TRDL?"))
        x0 = self.X0_DIVISION * tdiv + trdl

        # Get voltage scaling (dy, y0)
        dy = self._parse_numeric_response(
            self.query(f"C{channel}:VDIV?")
        ) * self.DY_ADC_CONVERSION
        y0 = -self._parse_numeric_response(self.query(f"C{channel}:OFFSET?"))

        return dx, x0, dy, y0

    def _get_trigger_time(self) -> str | None:
        """Get trigger timestamp."""
        trg_time = self.query("C1:INSPECT? TRIGGER_TIME")
        match = re.search(
            r"(?<=Time = )\s*\d+:\s*\d+:\s*\d+\.\d+", trg_time
        )
        return match.group(0) if match else None

    def _read_sequence_segments(self, channel: int) -> list[WaveformData]:
        """Read all sequence segments."""
        if not self._sequence_config:
            return []

        segments: list[WaveformData] = []
        dx, x0, dy, y0 = self._get_waveform_scaling(channel)

        for seg_idx in range(self._sequence_config.num_segments):
            self.write(f"C{channel}:WFSU SN,{seg_idx}")
            raw = self._read_channel_data(channel)

            segments.append(
                WaveformData(
                    raw_data=raw,
                    channel=channel,
                    segment=seg_idx,
                    dx=dx,
                    x0=x0,
                    dy=dy,
                    y0=y0,
                )
            )

        return segments


# === WaveRunner ===

class WaveRunner(WavePro):
    """Teledyne LeCroy WaveRunner series oscilloscope.

    Inherits from WavePro as SCPI commands are mostly identical.
    Only hardware specs differ.
    """

    MAX_SAMPLING_RATE: float = 40e9   # 40 GS/s
    MIN_MEMORY_SIZE: int = 500
    MAX_BANDWIDTH: float = 4e9        # 4 GHz
    ADC_BITS: int = 8

    # WaveRunner uses same SCPI commands as WavePro
    # Override methods here if commands differ

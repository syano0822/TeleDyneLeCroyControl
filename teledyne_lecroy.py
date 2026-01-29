#!/usr/bin/env python3
"""
Teledyne LeCroy oscilloscope library.

Supports WavePro and WaveRunner series oscilloscopes via VISA/TCP.

Bong-Hwi Lim (UTokyo)
"""
from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
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
    "Coupling",
    # Exceptions
    "ScopeError",
    "ScopeConnectionError",
    "ScopeTimeoutError",
    "ScopeConfigurationError",
    "ScopeTriggerError",
    # Config dataclasses
    "ChannelConfig",
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
    """Trigger edge direction."""
    RISING = auto()
    FALLING = auto()
    EITHER = auto()


class Coupling(Enum):
    """Channel input coupling/termination."""
    DC50 = "D50"   # 50Ω DC
    DC1M = "D1M"   # 1MΩ DC
    AC1M = "A1M"   # 1MΩ AC


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
class TriggerConfig:
    """Trigger configuration."""
    source_channels: list[int] = field(default_factory=lambda: [1])
    slope: TriggerSlope = TriggerSlope.FALLING
    level_offset: float = -0.01    # V relative to baseline
    mode: Literal["AUTO", "NORM", "SINGLE", "STOP"] = "SINGLE"


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
    ) -> None:
        """Configure scope with given settings."""
        self._validate_config(channels, acquisition)

        self.write("CHDR OFF")  # Remove response headers
        self._configure_display()
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

    # === Abstract Methods (implement in subclasses) ===

    @abstractmethod
    def _configure_display(self) -> None:
        """Configure display settings."""
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
        self._logger.info(
            f"Trigger configured: channels={config.source_channels}, "
            f"slope={config.slope.name}"
        )

    def arm(self, force: bool = False) -> None:
        """Arm trigger and wait for acquisition."""
        if force:
            self.write("TRMD NORM")
            self.write("FRTR")  # Force trigger
            self._logger.debug("Forced trigger")
        else:
            self.write("TRMD SINGLE")
            self.wait_opc()
            self._logger.debug("Armed, waiting for trigger")

    def is_triggered(self) -> bool:
        """Check if trigger has occurred."""
        return "STOP" in self.query("TRMD?")

    def wait_for_trigger(self, timeout: float | None = None) -> None:
        """Wait for trigger with optional timeout."""
        start = time.time()
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

    def _configure_display(self) -> None:
        """Configure display for remote operation."""
        self.write("DISP OFF")
        self.write("*RST")
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

    def _setup_trigger_source(self, config: TriggerConfig) -> None:
        """Setup trigger source with pattern trigger."""
        slope_map = {
            TriggerSlope.RISING: "H",
            TriggerSlope.FALLING: "L",
            TriggerSlope.EITHER: "X",
        }

        states = ["X"] * 4
        for ch in config.source_channels:
            states[ch - 1] = slope_map[config.slope]

        self.write(
            f"TRPA C1,{states[0]},C2,{states[1]},C3,{states[2]},C4,{states[3]},"
            "STATE,OR"
        )
        self.write("TRIG_SELECT PA,SR,C1")

    def _setup_trigger_level(self, config: TriggerConfig) -> None:
        """Setup trigger level relative to baseline."""
        for ch in config.source_channels:
            baseline = self._measure_baseline(ch)
            level = baseline + config.level_offset
            self.write(f"C{ch}:TRLV {level}")
            self._logger.debug(f"Channel {ch} trigger level: {level}")

    def _measure_baseline(self, channel: int) -> float:
        """Measure baseline using parameter measurement."""
        self.write(f"PACU 1,MEAN,C{channel}")

        # Trigger briefly to get measurement
        self.write("TRMD AUTO")
        time.sleep(0.5)
        self.write("TRMD NORM")

        baseline_str = self.query(
            r"""vbs? 'return=app.measure.p1.out.result.value' """
        ).split()[-1]

        return float(baseline_str)

    def _read_channel_data(self, channel: int) -> bytes:
        """Read raw waveform data."""
        self.write(f"C{channel}:WF? DAT1")
        return self.read_raw()

    def _get_waveform_scaling(
        self, channel: int
    ) -> tuple[float, float, float, float]:
        """Get waveform scaling factors."""
        if self._acquisition_config is None:
            raise ScopeConfigurationError("Acquisition not configured")

        dx = self._acquisition_config.sampling_period
        x0 = (
            self.X0_DIVISION * float(self.query("TDIV?"))
            + float(self.query("TRDL?"))
        )
        dy = float(self.query(f"C{channel}:VDIV?")) * self.DY_ADC_CONVERSION
        y0 = -float(self.query(f"C{channel}:OFFSET?"))

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

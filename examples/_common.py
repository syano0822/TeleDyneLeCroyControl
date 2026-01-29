from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from teledyne_lecroy import (
    AcquisitionConfig,
    ChannelConfig,
    ChannelTrigger,
    TriggerConfig,
    TriggerState,
    WavePro,
    WaveRunner,
    WaveformData,
)


ModelName = Literal["wavepro", "waverunner"]


def make_scope(model: ModelName, address: str):
    if model == "wavepro":
        return WavePro(address)
    if model == "waverunner":
        return WaveRunner(address)
    raise ValueError(f"Unknown model: {model}")


def default_channel_configs() -> dict[int, ChannelConfig]:
    return {
        1: ChannelConfig(vdiv=0.2, offset=0.0, enabled=True),
        2: ChannelConfig(vdiv=0.2, offset=0.0, enabled=True),
    }


def default_acquisition_config() -> AcquisitionConfig:
    return AcquisitionConfig(tdiv=1e-3, sampling_period=1e-6)


def default_trigger_config() -> TriggerConfig:
    return TriggerConfig(
        channels={
            1: ChannelTrigger(state=TriggerState.HIGH, level_offset=0.0),
            2: ChannelTrigger(state=TriggerState.HIGH, level_offset=0.0),
        },
        mode="SINGLE",
    )


def plot_waveform(waveform: WaveformData, output_path: str, title: str) -> None:
    t = waveform.to_time()
    v = waveform.to_voltage()
    print(
        f"CH{waveform.channel} stats: min={v.min():.3f} V, "
        f"max={v.max():.3f} V, mean={v.mean():.3f} V, p2p={(v.max()-v.min()):.3f} V"
    )

    plt.figure(figsize=(8, 4))
    plt.plot(t, v, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

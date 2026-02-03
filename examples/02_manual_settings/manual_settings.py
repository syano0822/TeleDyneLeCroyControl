#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from teledyne_lecroy import (
    AcquisitionConfig,
    AuxOutputMode,
    ChannelConfig,
    ChannelTrigger,
    Coupling,
    ScopeConfigurationError,
    ScopeConnectionError,
    SequenceConfig,
    TriggerConfig,
    TriggerState,
    WavePro,
    WaveRunner,
)


def make_scope(model: str, address: str):
    """Create a scope instance based on model name."""
    if model == "wavepro":
        return WavePro(address)
    if model == "waverunner":
        return WaveRunner(address)
    raise ValueError(f"Unknown model: {model}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["wavepro", "waverunner"], default="wavepro")
    p.add_argument("--address", default="192.168.0.10")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "settings.json",
        help="Path to JSON settings file",
    )
    return p.parse_args()


def load_settings(
    config_path: Path,
) -> tuple[dict[int, ChannelConfig], AcquisitionConfig, TriggerConfig, SequenceConfig, AuxOutputMode]:
    with open(config_path) as f:
        data = json.load(f)

    channels = {
        int(ch): ChannelConfig(
            vdiv=cfg["vdiv"],
            offset=cfg["offset"],
            coupling=Coupling[cfg.get("coupling", "DC50")],
            enabled=cfg["enabled"],
        )
        for ch, cfg in data["channels"].items()
    }

    acquisition = AcquisitionConfig(
        tdiv=data["acquisition"]["tdiv"],
        sampling_period=data["acquisition"]["sampling_period"],
        trigger_delay=data["acquisition"].get("trigger_delay", 0.0),
        window_delay=data["acquisition"].get("window_delay", 10e-9),
    )

    trigger_data = data["trigger"]
    trigger_channels = {}
    for ch, cfg in trigger_data.get("channels", {}).items():
        trigger_channels[int(ch)] = ChannelTrigger(
            state=TriggerState[cfg["state"]],
            level=cfg.get("level"),
            level_offset=cfg.get("level_offset", 0.0),
        )
    trigger = TriggerConfig(
        channels=trigger_channels,
        mode=trigger_data.get("mode", "SINGLE"),
        external=trigger_data.get("external", False),
        external_level=trigger_data.get("external_level", 1.25),
    )

    sequence_data = data.get("sequence", {})
    sequence = SequenceConfig(
        enabled=sequence_data.get("enabled", False),
        num_segments=sequence_data.get("num_segments", 1),
        timeout_enabled=sequence_data.get("timeout_enabled", False),
        timeout_seconds=sequence_data.get("timeout_seconds", 2.5e6),
    )

    aux_output_str = data.get("auxiliary_output", "TRIGGER_OUT")
    auxiliary_output = AuxOutputMode[aux_output_str]

    return channels, acquisition, trigger, sequence, auxiliary_output


def main() -> None:
    args = parse_args()
    try:
        print(f"Loading settings from: {args.config}")
        channels, acquisition, trigger, sequence, auxiliary_output = load_settings(args.config)

        with make_scope(args.model, args.address) as scope:
            scope.configure(channels=channels, acquisition=acquisition)
            scope.set_trigger(trigger)
            scope.configure_sequence(sequence)
            scope.set_auxiliary_output(auxiliary_output)

            print("Manual settings applied.")
            print(f"Channels: {list(channels.keys())}")
            print(f"TDIV: {acquisition.tdiv} s/div")
            trigger_info = ", ".join(
                f"CH{ch}={cfg.state.name}" for ch, cfg in trigger.channels.items()
            )
            print(f"Trigger: {trigger_info} ({trigger.mode})")
            print(f"Sequence: {'Enabled' if sequence.enabled else 'Disabled'} ({sequence.num_segments} segments)")
            print(f"Auxiliary Output: {auxiliary_output.name}")
            print("Verify changes on the scope front panel.")
    except (ScopeConnectionError, ScopeConfigurationError) as e:
        print(f"Configuration failed: {e}")
        raise


if __name__ == "__main__":
    main()

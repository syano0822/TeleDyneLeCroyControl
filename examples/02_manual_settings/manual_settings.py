#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from teledyne_lecroy import (
    AcquisitionConfig,
    ChannelConfig,
    ChannelTrigger,
    ScopeConfigurationError,
    ScopeConnectionError,
    TriggerConfig,
    TriggerState,
)
from examples._common import make_scope


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


def load_settings(config_path: Path) -> tuple[dict[int, ChannelConfig], AcquisitionConfig, TriggerConfig]:
    with open(config_path) as f:
        data = json.load(f)

    channels = {
        int(ch): ChannelConfig(
            vdiv=cfg["vdiv"],
            offset=cfg["offset"],
            enabled=cfg["enabled"],
        )
        for ch, cfg in data["channels"].items()
    }

    acquisition = AcquisitionConfig(
        tdiv=data["acquisition"]["tdiv"],
        sampling_period=data["acquisition"]["sampling_period"],
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

    return channels, acquisition, trigger


def main() -> None:
    args = parse_args()
    try:
        print(f"Loading settings from: {args.config}")
        channels, acquisition, trigger = load_settings(args.config)

        with make_scope(args.model, args.address) as scope:
            scope.configure(channels=channels, acquisition=acquisition)
            scope.set_trigger(trigger)

            print("Manual settings applied.")
            print(f"Channels: {list(channels.keys())}")
            print(f"TDIV: {acquisition.tdiv} s/div")
            trigger_info = ", ".join(
                f"CH{ch}={cfg.state.name}" for ch, cfg in trigger.channels.items()
            )
            print(f"Trigger: {trigger_info} ({trigger.mode})")
            print("Verify changes on the scope front panel.")
    except (ScopeConnectionError, ScopeConfigurationError) as e:
        print(f"Configuration failed: {e}")
        raise


if __name__ == "__main__":
    main()

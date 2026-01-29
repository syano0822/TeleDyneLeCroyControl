#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from teledyne_lecroy import (
    AcquisitionConfig,
    ChannelConfig,
    ScopeConfigurationError,
    ScopeConnectionError,
    TriggerConfig,
    TriggerSlope,
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

    trigger = TriggerConfig(
        source_channels=data["trigger"]["source_channels"],
        slope=TriggerSlope[data["trigger"]["slope"]],
        level_offset=data["trigger"]["level_offset"],
        mode=data["trigger"]["mode"],
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
            print(f"Trigger: {trigger.slope.name} edge on CH{trigger.source_channels}")
            print("Verify changes on the scope front panel.")
    except (ScopeConnectionError, ScopeConfigurationError) as e:
        print(f"Configuration failed: {e}")
        raise


if __name__ == "__main__":
    main()

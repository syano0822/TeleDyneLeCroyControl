#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from teledyne_lecroy import ScopeConnectionError, ScopeTimeoutError
from examples._common import make_scope, plot_waveform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["wavepro", "waverunner"], default="wavepro")
    p.add_argument("--address", default="192.168.0.10")
    p.add_argument("--outdir", default=".")
    p.add_argument(
        "--force",
        action="store_true",
        help="Use force trigger for testing without real signals",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        with make_scope(args.model, args.address) as scope:
            settings = scope.read_all_settings()
            enabled_channels = [
                int(ch)
                for ch, cfg in settings["channels"].items()
                if cfg.get("enabled")
            ]
            if not enabled_channels:
                enabled_channels = [1]

            scope.set_trigger_mode("SINGLE")

            print("Arming for single capture...")
            scope.arm()
            scope.wait_for_trigger(timeout=10.0, force=args.force)
            print("Triggered.")

            data = scope.readout(channels=enabled_channels)
            for ch, wf in data.items():
                plot_waveform(
                    wf,
                    str(outdir / f"single_capture_ch{ch}.png"),
                    title=f"Single Capture CH{ch}",
                )
                print(f"Saved plot: single_capture_ch{ch}.png")
    except (ScopeConnectionError, ScopeTimeoutError) as e:
        print(f"Capture failed: {e}")
        raise


if __name__ == "__main__":
    main()

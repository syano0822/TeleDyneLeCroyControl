#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from teledyne_lecroy import ScopeConnectionError, ScopeTimeoutError
from examples._common import (
    default_acquisition_config,
    default_channel_configs,
    default_trigger_config,
    make_scope,
    plot_waveform,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["wavepro", "waverunner"], default="wavepro")
    p.add_argument("--address", default="192.168.0.10")
    p.add_argument("--outdir", default=".")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        with make_scope(args.model, args.address) as scope:
            scope.configure(
                channels=default_channel_configs(),
                acquisition=default_acquisition_config(),
            )
            scope.set_trigger(default_trigger_config())

            print("Arming for single capture...")
            scope.arm()
            scope.wait_for_trigger(timeout=10.0)
            print("Triggered.")

            data = scope.readout()
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

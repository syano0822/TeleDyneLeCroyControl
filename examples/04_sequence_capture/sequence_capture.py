#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from teledyne_lecroy import ScopeConnectionError, ScopeTimeoutError, SequenceConfig
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
    p.add_argument("--segments", type=int, default=100)
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
            scope.configure(
                channels=default_channel_configs(),
                acquisition=default_acquisition_config(),
                sequence=SequenceConfig(enabled=True, num_segments=args.segments),
            )
            scope.set_trigger(default_trigger_config())

            print("Arming for sequence capture...")
            scope.arm()
            scope.wait_for_trigger(timeout=10.0, force=args.force)
            print("Triggered.")

            data = scope.readout_sequence()
            for ch, seq in data.items():
                print(f"CH{ch} segments: {len(seq)}")
                # Plot first 3 segments as a sanity check
                for idx in range(min(3, len(seq))):
                    wf = seq[idx]
                    plot_waveform(
                        wf,
                        str(outdir / f"sequence_ch{ch}_seg{idx}.png"),
                        title=f"Sequence CH{ch} Seg {idx}",
                    )
                    print(f"Saved plot: sequence_ch{ch}_seg{idx}.png")
    except (ScopeConnectionError, ScopeTimeoutError) as e:
        print(f"Sequence capture failed: {e}")
        raise


if __name__ == "__main__":
    main()

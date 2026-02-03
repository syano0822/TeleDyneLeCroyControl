#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no X window needed)
import matplotlib.pyplot as plt

# Add parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from teledyne_lecroy import ScopeConnectionError, ScopeTimeoutError, SequenceData, WaveformData, WavePro, WaveRunner


def make_scope(model: str, address: str):
    """Create a scope instance based on model name."""
    if model == "wavepro":
        return WavePro(address)
    if model == "waverunner":
        return WaveRunner(address)
    raise ValueError(f"Unknown model: {model}")


def plot_waveform(waveform: WaveformData, output_path: str, title: str) -> None:
    """Plot and save waveform data as PNG."""
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


def timed(name: str):
    """Context manager to measure and print execution time."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f"  [{name}] {elapsed*1000:.1f} ms")
    return Timer()


def save_sequence(seq: SequenceData, filepath: Path) -> None:
    """Save sequence data to numpy .npz file."""
    # Use the library method which handles inhomogeneous segment lengths (trimming)
    voltages = seq.to_voltage_array()
    
    # Calculate time axis for one segment (assuming uniform)
    # We take the first segment as reference
    if len(seq) > 0:
        time_axis = seq[0].to_time()
        # Ensure time axis matches voltage array width (in case of trimming)
        if len(time_axis) > voltages.shape[1]:
            time_axis = time_axis[:voltages.shape[1]]
    else:
        time_axis = np.array([])

    np.savez_compressed(
        filepath,
        voltages=voltages,  # Shape: (n_segments, n_points)
        time=time_axis,
        channel=seq.channel,
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


def load_or_create_settings(scope, settings_file: Path) -> dict:
    """Load settings from file or create from current scope state."""
    if settings_file.exists():
        print(f"Loading settings from {settings_file}")
        return scope.load_settings_file(settings_file)
    else:
        print("No settings file found, reading current scope settings...")
        settings = scope.read_all_settings()
        scope.save_settings(settings_file)
        print(f"Settings saved to {settings_file}")
        return settings


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    settings_file = outdir / "settings.json"

    try:
        with make_scope(args.model, args.address) as scope:
            with timed("load_settings"):
                settings = load_or_create_settings(scope, settings_file)

            # Override sequence settings with command line args
            settings["sequence"] = {
                "enabled": True,
                "num_segments": args.segments,
                "timeout_enabled": settings.get("sequence", {}).get("timeout_enabled", False),
                "timeout_seconds": settings.get("sequence", {}).get("timeout_seconds", 2.5e6),
            }

            with timed("apply_settings"):
                scope.apply_settings(settings)

            enabled_channels = [
                int(ch)
                for ch, cfg in settings["channels"].items()
                if cfg.get("enabled")
            ]
            if not enabled_channels:
                enabled_channels = [1]

            print(f"Arming for sequence capture ({args.segments} segments)...")
            with timed("arm"):
                scope.arm()

            with timed("wait_for_trigger"):
                scope.wait_for_trigger(timeout=10.0, force=args.force)
            print("Triggered.")

            with timed("readout_sequence"):
                data = scope.readout_sequence(channels=enabled_channels)

            # Data size diagnostics
            print("\n=== Data Size ===")
            total_bytes = 0
            for ch, seq in data.items():
                ch_bytes = sum(len(seg.raw_data) for seg in seq.segments)
                total_bytes += ch_bytes
                points_per_seg = len(seq[0].raw_data) if len(seq) > 0 else 0
                print(f"  CH{ch}: {len(seq)} segments × {points_per_seg:,} points = {ch_bytes:,} bytes ({ch_bytes/1e6:.2f} MB)")
            print(f"  Total: {total_bytes:,} bytes ({total_bytes/1e6:.2f} MB)")
            print()

            for ch, seq in data.items():
                print(f"CH{ch}: {len(seq)} segments")

                # Save all sequence data
                data_file = outdir / f"sequence_ch{ch}.npz"
                with timed(f"save_data_ch{ch}"):
                    save_sequence(seq, data_file)
                print(f"Saved data: {data_file.name}")

                # Plot first 3 segments as a sanity check
                for idx in range(min(3, len(seq))):
                    wf = seq[idx]
                    with timed(f"plot_ch{ch}_seg{idx}"):
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

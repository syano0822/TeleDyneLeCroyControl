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

from teledyne_lecroy import ScopeConnectionError, ScopeTimeoutError, WaveformData, WavePro, WaveRunner


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


def save_waveform(wf: WaveformData, filepath: Path) -> None:
    """Save waveform data to numpy .npz file."""
    np.savez_compressed(
        filepath,
        time=wf.to_time(),
        voltage=wf.to_voltage(),
        raw=np.frombuffer(wf.raw_data, dtype=np.int8),
        dx=wf.dx,
        x0=wf.x0,
        dy=wf.dy,
        y0=wf.y0,
        channel=wf.channel,
        segment=wf.segment,
    )


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

            # Force disable sequence mode for single capture
            settings["sequence"] = {"enabled": False}

            with timed("apply_settings"):
                scope.apply_settings(settings)

            enabled_channels = [
                int(ch)
                for ch, cfg in settings["channels"].items()
                if cfg.get("enabled")
            ]
            if not enabled_channels:
                enabled_channels = [1]

            print("Arming for single capture...")
            with timed("arm"):
                scope.arm()

            with timed("wait_for_trigger"):
                scope.wait_for_trigger(timeout=10.0, force=args.force)
            print("Triggered.")

            with timed("readout"):
                data = scope.readout(channels=enabled_channels)

            # Data size diagnostics
            print("\n=== Data Size ===")
            total_bytes = 0
            for ch, wf in data.items():
                raw_bytes = len(wf.raw_data)
                total_bytes += raw_bytes
                print(f"  CH{ch}: {raw_bytes:,} bytes ({raw_bytes/1e6:.2f} MB), {raw_bytes:,} points")
            print(f"  Total: {total_bytes:,} bytes ({total_bytes/1e6:.2f} MB)")
            print()

            for ch, wf in data.items():
                # Save data
                data_file = outdir / f"single_capture_ch{ch}.npz"
                with timed(f"save_data_ch{ch}"):
                    save_waveform(wf, data_file)
                print(f"Saved data: {data_file.name}")

                # Save plot
                with timed(f"plot_ch{ch}"):
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from teledyne_lecroy import ScopeConnectionError, WavePro, WaveRunner


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with make_scope(args.model, args.address) as scope:
            idn = scope.query("*IDN?")
            print(f"Connected ({args.model}) -> {idn}")
    except ScopeConnectionError as e:
        print(f"Connection failed: {e}")
        raise


if __name__ == "__main__":
    main()

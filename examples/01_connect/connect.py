#!/usr/bin/env python3
from __future__ import annotations

import argparse

from teledyne_lecroy import ScopeConnectionError
from examples._common import make_scope


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

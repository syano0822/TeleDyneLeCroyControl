#!/usr/bin/env python3
# scope_snapshot.py
#
# Snapshot LeCroy scope settings:
#  - Print human-readable summary (optional)
#  - Dump JSON snapshot file (optional)
#  - Show missing/unsupported keys (optional)
#  - Diff against a previous snapshot JSON (optional)
#
# Added:
#  - IP -> NAME mapping from config.json (scope_profile.scope_names)
#  - Output filename: snapshots/<MODEL>_<NAME>_<runXXXXXX>.json (fallback to timestamp)

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lecroy_visa import LeCroyVisa

INVALID_MARKERS = (
    "Object doesn't support this property or method",
    "Invalid procedure call or argument",
)


# -----------------------------
# Helpers: JSON I/O
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sanitize_name(name: str) -> str:
    name = name.strip()
    if not name:
        return "NONAME"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def resolve_scope_name(cfg: Dict[str, Any], ip: str) -> str:
    mapping = cfg.get("scope_profile", {}).get("scope_names", {})
    if isinstance(mapping, dict):
        v = mapping.get(ip)
        if v is not None and str(v).strip():
            return sanitize_name(str(v))
    return f"IP_{ip.replace('.', '_')}"


def extract_run_tag_from_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    m = re.search(r"(run\d{6})", str(title))
    if m:
        return m.group(1)
    return None


# -----------------------------
# Helpers: VBS read with robust unsupported detection
# -----------------------------
def is_unsupported(raw: Optional[str]) -> Tuple[bool, str]:
    if raw is None:
        return True, "no return"
    r = str(raw).strip()
    if r == "":
        return True, "empty return"
    for m in INVALID_MARKERS:
        if m in r:
            return True, r
    return False, ""


def safe_read_key(scope: LeCroyVisa, key: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    try:
        vbs_path = scope.to_vbs_path(key)
    except Exception as e:
        return None, {"key": key, "path": "(unresolved)", "reason": f"{type(e).__name__}: {e}"}

    try:
        raw = scope.vbs_get(vbs_path)
        bad, reason = is_unsupported(raw)
        if bad:
            return None, {"key": key, "path": vbs_path, "reason": reason}
        return str(raw).strip(), None
    except Exception as e:
        return None, {"key": key, "path": vbs_path, "reason": f"{type(e).__name__}: {e}"}


# -----------------------------
# Snapshot model
# -----------------------------
@dataclass
class Snapshot:
    idn: str
    model: str
    family: str
    ip: str
    scope_name: str
    run_tag: Optional[str]
    timestamp_local: str
    readback_human: Dict[str, Any]
    readback_raw: Dict[str, Any]
    missing: List[Dict[str, str]]

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "idn": self.idn,
            "model": self.model,
            "family": self.family,
            "ip": self.ip,
            "scope_name": self.scope_name,
            "run_tag": self.run_tag,
            "timestamp_local": self.timestamp_local,
            "readback_human": self.readback_human,
            "missing": self.missing,
        }
        if include_raw:
            d["readback_raw"] = self.readback_raw
        return d


# -----------------------------
# Build target key list
# -----------------------------
def get_channels_max(cfg: Dict[str, Any], override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    return int(cfg.get("scope_profile", {}).get("channels_max", 8))


def build_keys(cfg: Dict[str, Any], channels_max: int, only: str) -> List[str]:
    only = only.lower()
    include_all = (only == "all")

    keys: List[str] = []

    # Horizontal
    if include_all or only == "horizontal":
        keys += [
            "horizontal.sample_mode",
            "horizontal.num_segments",
            "horizontal.sequence_timeout_enable",
            "horizontal.sequence_timeout",
            "horizontal.hor_scale",
            "horizontal.hor_offset",
            "horizontal.num_points",
            "horizontal.time_per_point",
        ]

    # Trigger
    if include_all or only == "trigger":
        keys += [
            "trigger.type",
            "trigger.source",
            "trigger.pattern_type",
            "trigger.pattern_level",
        ]
        for src in [f"C{i}" for i in range(1, channels_max + 1)] + ["Ext"]:
            keys.append(f"trigger.levels.{src}")
            keys.append(f"trigger.slopes.{src}")
            keys.append(f"trigger.pattern_states.{src}")

    # Channels
    if include_all or only == "channels":
        ch_fields = ["view", "coupling", "scale", "offset", "invert", "bandwidth_limit", "deskew", "units"]
        for i in range(1, channels_max + 1):
            ch = f"C{i}"
            for f in ch_fields:
                keys.append(f"channels.{ch}.{f}")

    # Save waveform settings (properties only)
    if include_all or only in ("save", "save_waveforms"):
        keys += [
            "save.save_to",
            "save.waveform_dir",
            "save.wave_format",
            "save.trace_title",   # <- resolved prefix is expected here if apply_config set it
            "save.save_source",
        ]

    # AUX OUT
    if include_all or only in ("aux", "aux_out"):
        keys += [
            "aux_out.mode",
            "aux_out.aux_in_coupling",
        ]

    return keys


# -----------------------------
# Pretty printing helpers
# -----------------------------
def _fmt_kv(k: str, v: Any, pad: int) -> str:
    s = "" if v is None else str(v)
    return f"{k:<{pad}} : {s}"


def print_header(scope: LeCroyVisa, ip: str, scope_name: str) -> None:
    print(f"Connected: {scope.idn()}")
    print(f"Model    : {scope.model}   Family: {scope.family}")
    print(f"IP       : {ip}")
    print(f"Name     : {scope_name}")


def print_group(title: str, items: List[Tuple[str, Any]]) -> None:
    if not items:
        return
    print(f"\n[{title}]")
    pad = max(len(k) for k, _ in items)
    for k, v in items:
        print(_fmt_kv(k, v, pad))


def _norm_trigger_type(raw: Any) -> str:
    s = ("" if raw is None else str(raw)).strip().lower()
    if s == "edge":
        return "EDGE"
    if s in ("logic", "pattern"):
        return "LOGIC"
    return "OTHER"


def group_items(readback_human: Dict[str, Any], only: str, trigger_view: str) -> Dict[str, List[Tuple[str, Any]]]:
    only = only.lower()
    trigger_view = trigger_view.lower()

    groups: Dict[str, List[Tuple[str, Any]]] = {
        "Trigger": [],
        "Horizontal": [],
        "Channels": [],
        "Save": [],
        "AUX OUT": [],
    }

    def add(group: str, key: str, label: str) -> None:
        if key in readback_human:
            groups[group].append((label, readback_human.get(key)))

    # Horizontal
    if only in ("all", "horizontal"):
        add("Horizontal", "horizontal.sample_mode", "sample_mode")
        add("Horizontal", "horizontal.num_segments", "num_segments")
        add("Horizontal", "horizontal.sequence_timeout_enable", "sequence_timeout_enable")
        add("Horizontal", "horizontal.sequence_timeout", "sequence_timeout")
        add("Horizontal", "horizontal.hor_scale", "hor_scale")
        add("Horizontal", "horizontal.hor_offset", "hor_offset")
        add("Horizontal", "horizontal.num_points", "num_points")
        add("Horizontal", "horizontal.time_per_point", "time_per_point")

    # Trigger
    if only in ("all", "trigger"):
        t_raw = readback_human.get("trigger.type")
        t_norm = _norm_trigger_type(t_raw)
        src = readback_human.get("trigger.source")

        if trigger_view == "full":
            add("Trigger", "trigger.type", "type")
            add("Trigger", "trigger.source", "source")
            add("Trigger", "trigger.pattern_type", "pattern_type")
            add("Trigger", "trigger.pattern_level", "pattern_level")

            for k in sorted(readback_human.keys()):
                if k.startswith("trigger.levels."):
                    add("Trigger", k, k.replace("trigger.levels.", "level."))
            for k in sorted(readback_human.keys()):
                if k.startswith("trigger.slopes."):
                    add("Trigger", k, k.replace("trigger.slopes.", "slope."))
            for k in sorted(readback_human.keys()):
                if k.startswith("trigger.pattern_states."):
                    add("Trigger", k, k.replace("trigger.pattern_states.", "pstate."))

        else:
            # SMART:
            # - EDGE: show only source-related
            # - LOGIC: show all (including Don't Care) for safety
            add("Trigger", "trigger.type", "type")

            if t_norm == "EDGE":
                add("Trigger", "trigger.source", "source")
                if src is not None:
                    k_level = f"trigger.levels.{src}"
                    k_slope = f"trigger.slopes.{src}"
                    if k_level in readback_human:
                        add("Trigger", k_level, f"level.{src}")
                    if k_slope in readback_human:
                        add("Trigger", k_slope, f"slope.{src}")

            elif t_norm == "LOGIC":
                add("Trigger", "trigger.pattern_type", "pattern_type")
                add("Trigger", "trigger.pattern_level", "pattern_level")
                add("Trigger", "trigger.source", "source")

                for k in sorted(readback_human.keys()):
                    if k.startswith("trigger.pattern_states."):
                        add("Trigger", k, k.replace("trigger.pattern_states.", "pstate."))
                for k in sorted(readback_human.keys()):
                    if k.startswith("trigger.levels."):
                        add("Trigger", k, k.replace("trigger.levels.", "level."))
                for k in sorted(readback_human.keys()):
                    if k.startswith("trigger.slopes."):
                        add("Trigger", k, k.replace("trigger.slopes.", "slope."))

            else:
                add("Trigger", "trigger.source", "source")
                add("Trigger", "trigger.pattern_type", "pattern_type")
                add("Trigger", "trigger.pattern_level", "pattern_level")
                for k in sorted(readback_human.keys()):
                    if k.startswith("trigger.levels."):
                        add("Trigger", k, k.replace("trigger.levels.", "level."))
                for k in sorted(readback_human.keys()):
                    if k.startswith("trigger.slopes."):
                        add("Trigger", k, k.replace("trigger.slopes.", "slope."))
                for k in sorted(readback_human.keys()):
                    if k.startswith("trigger.pattern_states."):
                        add("Trigger", k, k.replace("trigger.pattern_states.", "pstate."))

    # Channels
    if only in ("all", "channels"):
        for k in sorted(readback_human.keys()):
            if k.startswith("channels."):
                groups["Channels"].append((k.replace("channels.", ""), readback_human.get(k)))

    # Save
    if only in ("all", "save", "save_waveforms"):
        add("Save", "save.save_to", "save_to")
        add("Save", "save.waveform_dir", "waveform_dir")
        add("Save", "save.wave_format", "wave_format")
        add("Save", "save.trace_title", "trace_title")
        add("Save", "save.save_source", "save_source")

    # AUX OUT
    if only in ("all", "aux", "aux_out"):
        add("AUX OUT", "aux_out.mode", "mode")
        add("AUX OUT", "aux_out.aux_in_coupling", "aux_in_coupling")

    return groups


def print_missing(missing: List[Dict[str, str]], show_all: bool, limit: int = 50) -> None:
    if not missing:
        print("\n[Missing] none")
        return

    print(f"\n[Missing] {len(missing)} key(s) unavailable/failed")
    items = missing if show_all else missing[:limit]
    for m in items:
        print(f"  - {m['key']} ({m['path']}): {m['reason']}")
    if (not show_all) and len(missing) > limit:
        print(f"  ... ({len(missing)-limit} more; use --show-missing for all)")


# -----------------------------
# Diff
# -----------------------------
def diff_snapshots(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    old_rb = old.get("readback_human", {}) or {}
    new_rb = new.get("readback_human", {}) or {}

    added: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}
    changed: Dict[str, Dict[str, Any]] = {}

    old_keys = set(old_rb.keys())
    new_keys = set(new_rb.keys())

    for k in sorted(new_keys - old_keys):
        added[k] = new_rb.get(k)

    for k in sorted(old_keys - new_keys):
        removed[k] = old_rb.get(k)

    for k in sorted(old_keys & new_keys):
        if old_rb.get(k) != new_rb.get(k):
            changed[k] = {"old": old_rb.get(k), "new": new_rb.get(k)}

    return {"added": added, "removed": removed, "changed": changed}


def print_diff(d: Dict[str, Any]) -> None:
    added = d["added"]
    removed = d["removed"]
    changed = d["changed"]

    print("\n=== Diff (readback_human) ===")
    if not added and not removed and not changed:
        print("No differences.")
        return

    if added:
        print(f"\n[Added] {len(added)}")
        for k, v in added.items():
            print(f"  + {k} : {v}")

    if removed:
        print(f"\n[Removed] {len(removed)}")
        for k, v in removed.items():
            print(f"  - {k} : {v}")

    if changed:
        print(f"\n[Changed] {len(changed)}")
        for k, vv in changed.items():
            print(f"  * {k}")
            print(f"      old: {vv['old']}")
            print(f"      new: {vv['new']}")


# -----------------------------
# Output path builder
# -----------------------------
def build_out_path(
    out_arg: Optional[str],
    model: str,
    scope_name: str,
    run_tag: Optional[str],
) -> Optional[Path]:
    """
    If out_arg is:
      - None: return None
      - ends with '.json': treat as file path (no auto naming)
      - otherwise: treat as directory, create auto file name
    """
    if out_arg is None:
        return None

    p = Path(out_arg)

    # Explicit file path
    if p.suffix.lower() == ".json":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # Directory mode
    p.mkdir(parents=True, exist_ok=True)

    if run_tag:
        fname = f"{model}_{scope_name}_{run_tag}.json"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{model}_{scope_name}_{ts}.json"

    return p / fname


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot LeCroy scope settings (print + JSON dump + diff).")
    ap.add_argument("--ip", required=True, help="Scope IP address")
    ap.add_argument("--backend", default="@py", help="pyvisa backend (default: @py)")
    ap.add_argument("--timeout", type=float, default=5.0, help="VISA timeout seconds")

    ap.add_argument("--config", default="config.json", help="Config JSON path (used for channels_max and IP->NAME)")
    ap.add_argument(
        "--channels-max",
        type=int,
        default=None,
        help="Override scope_profile.channels_max (e.g. 8 for WR, 4 for WP)",
    )

    ap.add_argument(
        "--only",
        default="all",
        choices=["all", "trigger", "channels", "horizontal", "save", "save_waveforms", "aux", "aux_out"],
        help="Limit snapshot to a category",
    )

    ap.add_argument(
        "--trigger-view",
        default="smart",
        choices=["smart", "full"],
        help="Trigger print style: smart (Edge minimal, Logic full) or full (always all trigger keys)",
    )

    ap.add_argument("--print", dest="do_print", action="store_true", help="Print human-readable snapshot")
    ap.add_argument("--no-print", dest="do_print", action="store_false", help="Do not print snapshot")
    ap.set_defaults(do_print=True)

    ap.add_argument("--json", dest="do_json", action="store_true", help="Write JSON snapshot")
    ap.add_argument("--no-json", dest="do_json", action="store_false", help="Do not write JSON snapshot")
    ap.set_defaults(do_json=True)

    ap.add_argument(
        "--out",
        default="snapshots/",
        help="Output directory (default: snapshots/) or explicit .json file path.",
    )

    ap.add_argument("--include-raw", action="store_true", help="Include readback_raw in JSON (bigger)")
    ap.add_argument("--no-raw", action="store_true", help="Exclude readback_raw from JSON (smaller)")

    ap.add_argument("--show-missing", action="store_true", help="Print all missing/unsupported keys")
    ap.add_argument("--strict", action="store_true", help="Exit with non-zero status if any missing keys")
    ap.add_argument("--diff", default=None, help="Diff against a previous snapshot JSON file")

    args = ap.parse_args()

    cfg = load_json(args.config)
    channels_max = get_channels_max(cfg, args.channels_max)

    scope_name = resolve_scope_name(cfg, args.ip)

    scope = LeCroyVisa(address=args.ip, visa_backend=args.backend, timeout_s=args.timeout)
    scope.connect()

    keys = build_keys(cfg, channels_max=channels_max, only=args.only)

    readback_human: Dict[str, Any] = {}
    readback_raw: Dict[str, Any] = {}
    missing: List[Dict[str, str]] = []

    for k in keys:
        val, miss = safe_read_key(scope, k)
        readback_human[k] = val

        # Raw mapping by resolved VBS path
        try:
            vbs_path = scope.to_vbs_path(k)
            readback_raw[vbs_path] = val
        except Exception:
            pass

        if miss is not None:
            missing.append(miss)

    run_tag = extract_run_tag_from_title(readback_human.get("save.trace_title"))

    snap = Snapshot(
        idn=scope.idn(),
        model=scope.model,
        family=scope.family,
        ip=args.ip,
        scope_name=scope_name,
        run_tag=run_tag,
        timestamp_local=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        readback_human=readback_human,
        readback_raw=readback_raw,
        missing=missing,
    )

    # Print
    if args.do_print:
        print_header(scope, args.ip, scope_name)

        if run_tag:
            print(f"Run      : {run_tag}")

        groups = group_items(
            readback_human=readback_human,
            only=args.only,
            trigger_view=args.trigger_view,
        )
        for title in ["Trigger", "Horizontal", "AUX OUT", "Save", "Channels"]:
            print_group(title, groups.get(title, []))

        print_missing(missing, show_all=args.show_missing)

        if args.diff:
            old = load_json(args.diff)
            d = diff_snapshots(old, snap.to_dict(include_raw=True))
            print_diff(d)

    # JSON dump
    if args.do_json:
        include_raw = args.include_raw and (not args.no_raw)
        if args.no_raw:
            include_raw = False

        out_path = build_out_path(
            args.out,
            model=scope.model,
            scope_name=scope_name,
            run_tag=run_tag,
        )
        if out_path is not None:
            write_json(out_path, snap.to_dict(include_raw=include_raw))
            if args.do_print:
                print(f"\nSaved JSON: {out_path}")

    scope.close()

    if args.strict and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""
Finding recordings under INPUT_DIR, by name and content, recursively, following symlinks.

A *source* is one recording on disk in one format (a SpikeGLX run, an Open Ephys recording folder,
an Intan file, an NWB file, a SpikeInterface folder, or a raw binary file with a JSON sidecar). A
source holds one or more *streams* (one per probe); every stream of a source becomes one session.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from processor.common import SAFE_ID, log

BINARY_EXT = {".bin", ".dat", ".raw"}
SKIP_DIRS = {"__pycache__", ".git", "properties"}
SUB_SES = re.compile(r"(sub-[A-Za-z0-9-]+)_(ses-[A-Za-z0-9-]+)")
UUID_HEAD = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
SPIKEGLX_AP = re.compile(r"^(?P<run>.+?)_g(?P<gate>\d+)(_t(?P<trig>\d+|cat))?\.imec(?P<probe>\d*)\.ap\.c?bin$")


@dataclass
class Source:
    kind: str                 # spikeglx | cbin | openephys | intan | nwb | si_folder | binary
    path: Path                # the file or folder handed to the reader
    name: str                 # the name a session id is derived from (file stem or folder name)
    files: list[Path] = field(default_factory=list)   # what was matched (for logs and provenance)
    probe_json: Path | None = None    # a probeinterface JSON next to it (used when the format has no geometry)
    sidecar: Path | None = None       # binary fallback: the JSON with sampling rate, channels, dtype


def _walk(root: Path):
    """os.walk following symlinks, guarding against loops; yields (dir, subdirs, files) as Paths."""
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
        yield Path(dirpath), dirnames, sorted(f for f in filenames if not f.startswith("."))


def is_probeinterface_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        if path.stat().st_size > 50_000_000:
            return False
        with open(path) as fh:
            head = fh.read(4096)
        return '"specification"' in head and "probeinterface" in head
    except OSError:
        return False


def _binary_sidecar(path: Path) -> Path | None:
    for cand in (path.with_name(path.name + ".json"), path.with_suffix(".json")):
        if cand.is_file() and not is_probeinterface_json(cand):
            try:
                meta = json.loads(cand.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict) and "sampling_frequency" in meta and "num_channels" in meta:
                return cand
    return None


def _probe_json_for(folder: Path, stem: str | None = None) -> Path | None:
    cands = sorted(p for p in folder.glob("*.json") if is_probeinterface_json(p))
    if stem:
        named = [p for p in cands if p.name.startswith(stem)]
        if named:
            return named[0]
    return cands[0] if len(cands) == 1 else (cands[0] if cands else None)


def find_sources(input_dir: Path) -> tuple[list[Source], Path | None]:
    """All recordings under input_dir, and the sessions.csv sheet if there is one."""
    sources: list[Source] = []
    sheet: Path | None = None
    claimed: set[str] = set()          # real paths already part of a source
    spikeglx_runs: dict[str, Source] = {}

    def claim(p: Path):
        claimed.add(os.path.realpath(p))

    def inside_claimed(p: Path) -> bool:
        rp = os.path.realpath(p)
        return any(rp == c or rp.startswith(c + os.sep) for c in claimed)

    for d, dirs, files in _walk(input_dir):
        if inside_claimed(d):
            dirs[:] = []
            continue
        names = set(files)
        # A SpikeInterface folder (binary_folder or a zarr recording): read whole, don't descend.
        if "si_folder.json" in names or (d.suffix == ".zarr" and ".zattrs" in names and _si_zarr(d)):
            if "si_folder.json" in names and _si_folder_is_recording(d):
                sources.append(Source("si_folder", d, d.name, [d / "si_folder.json"]))
            elif d.suffix == ".zarr":
                sources.append(Source("si_folder", d, d.name.removesuffix(".zarr"), [d]))
            claim(d)
            dirs[:] = []
            continue
        # Open Ephys: a folder holding "Record Node ..." folders, or a Record Node / legacy folder itself.
        record_nodes = [x for x in dirs if x.startswith("Record Node")]
        if record_nodes or "settings.xml" in names and ("structure.oebin" in names or any(f.endswith(".continuous") for f in names)
                                                         or _has_oebin_below(d)):
            root = d
            name = d.name
            if not record_nodes and d.name.startswith("Record Node"):
                name = d.parent.name
            sources.append(Source("openephys", root, name, [root]))
            claim(root)
            dirs[:] = []
            continue
        for f in files:
            p = d / f
            low = f.lower()
            if f == "sessions.csv":
                if sheet is None:
                    sheet = p
                continue
            m = SPIKEGLX_AP.match(f)
            if m and (d / (f.rsplit(".", 1)[0] + ".meta")).exists():
                if low.endswith(".cbin"):
                    sources.append(Source("cbin", d, m.group("run") + f"_g{m.group('gate')}", [p]))
                    claim(p)
                    continue
                # One source per SpikeGLX run folder: <run>_g0/ holding <run>_g0_imec0/, _imec1/ ...
                run_dir = d.parent if re.search(r"_imec\d+$", d.name) and d.parent.name.startswith(m.group("run")) else d
                key = os.path.realpath(run_dir)
                run_name = f"{m.group('run')}_g{m.group('gate')}"
                if key not in spikeglx_runs:
                    spikeglx_runs[key] = Source("spikeglx", run_dir, run_name, [])
                    sources.append(spikeglx_runs[key])
                spikeglx_runs[key].files.append(p)
                claim(p)
                continue
            if low.endswith((".rhd", ".rhs")):
                if f in ("info.rhd", "info.rhs"):      # one-file-per-signal / per-channel layout: the folder
                    sources.append(Source("intan", p, d.name, [p], probe_json=_probe_json_for(d)))
                    claim(d)
                    break
                sources.append(Source("intan", p, p.stem, [p], probe_json=_probe_json_for(d, p.stem)))
                claim(p)
                continue
            if low.endswith(".nwb"):
                sources.append(Source("nwb", p, p.stem, [p], probe_json=_probe_json_for(d, p.stem)))
                claim(p)
                continue
        # Raw binary fallback: <name>.bin|.dat|.raw + <name>.json sidecar (+ a probeinterface JSON).
        for f in files:
            p = d / f
            if p.suffix.lower() in BINARY_EXT and not inside_claimed(p) and not f.endswith((".ap.bin", ".lf.bin")):
                sc = _binary_sidecar(p)
                if sc is None:
                    log.info("skipping %s: no <name>.json sidecar with sampling_frequency and num_channels", p)
                    continue
                sources.append(Source("binary", p, p.stem, [p, sc], probe_json=_probe_json_for(d, p.stem), sidecar=sc))
                claim(p)
    return sources, sheet


def _si_folder_is_recording(d: Path) -> bool:
    try:
        meta = json.loads((d / "si_folder.json").read_text())
        return "Recording" in str(meta.get("object", "")) or (d / "binary.json").exists() and "Sorting" not in str(meta.get("object", ""))
    except (OSError, ValueError):
        return False


def _si_zarr(d: Path) -> bool:
    try:
        att = json.loads((d / ".zattrs").read_text())
        return "spikeinterface" in json.dumps(att)[:20000].lower() and "sampling_frequency" in att
    except (OSError, ValueError):
        return False


def _has_oebin_below(d: Path) -> bool:
    for _, _, files in os.walk(d, followlinks=True):
        if "structure.oebin" in files:
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Sessions


def read_sheet(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    out = []
    for r in rows:
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        if not r.get("session_id"):
            continue
        r["session_id"] = SAFE_ID.sub("-", r["session_id"])
        out.append(r)
    return out


def match_sheet(src: Source, rows: list[dict], input_dir: Path) -> dict | None:
    """The sheet row whose `path` names this source's file or folder, or one of its parent folders."""
    try:
        rel = src.path.relative_to(input_dir)
        parts = list(rel.parts)
    except ValueError:
        parts = [src.path.name]
    names = set(parts) | {src.name, src.path.stem}
    rel_s = "/".join(parts)
    for r in rows:
        p = (r.get("path") or "").strip().strip("/")
        if not p:
            continue
        if p in names or rel_s == p or rel_s.endswith("/" + p) or rel_s.startswith(p + "/"):
            return r
    return None


def session_id_for(src: Source, input_dir: Path) -> str:
    """sub-<a>_ses-<b> from the name or a parent folder (UUID session ids cut to 8 characters), else the name."""
    try:
        parents = list(src.path.relative_to(input_dir).parts)
    except ValueError:
        parents = [src.path.name]
    for name in [src.path.name] + parents[::-1]:
        m = SUB_SES.search(name)
        if m:
            sub, ses = m.group(1), m.group(2)[4:]
            if UUID_HEAD.match(ses):
                ses = ses[:8]
            return f"{sub}_ses-{ses}"
    return SAFE_ID.sub("-", src.name) or "session"


def probe_suffix(stream_label: str, index: int) -> str:
    """'_probe00' from a stream name such as imec0.ap, ElectricalSeriesProbe01AP, ProbeA-AP; else the index."""
    m = re.search(r"imec(\d+)", stream_label, re.I) or re.search(r"probe[_ -]?(\d+)", stream_label, re.I)
    if m:
        return f"_probe{int(m.group(1)):02d}"
    m = re.search(r"[Pp]robe[_ -]?([A-Z])(?![a-z])", stream_label)
    if m:
        return f"_probe{ord(m.group(1)) - ord('A'):02d}"
    return f"_probe{index:02d}"

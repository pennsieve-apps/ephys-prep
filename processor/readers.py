"""
Opening a source with SpikeInterface: one lazy recording per stream (probe), with its probe geometry,
session-clock times and what we know about the probe and the source file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from processor.common import log
from processor.inputs import Source

LFP_MAX_RATE = 10_000.0   # streams sampled below this are LFP / auxiliary, not spike-band


@dataclass
class Stream:
    label: str                     # stream name in the source (imec0.ap, ElectricalSeriesProbe00AP, ...)
    open: object                   # callable -> BaseRecording (lazy)
    probe: dict = field(default_factory=dict)   # model, serial, part_number, geometry_source
    time_source: str = "t_start + n / rate"
    nwb: dict | None = None        # source NWB identity (session_start_time, identifier, ...)
    timestamps: object = None      # callable -> float64 array of explicit sample times (NWB `timestamps`), else None
    lfp: object = None             # callable -> (LFP recording of the same probe, its timestamps or None), else None


def list_streams(src: Source) -> list[Stream]:
    k = src.kind
    if k == "spikeglx":
        return _spikeglx_streams(src)
    if k == "cbin":
        return [Stream(src.files[0].name.split(".")[-3] + ".ap", lambda: _open_cbin(src), _spikeglx_probe_meta(src.files[0]),
                       "SpikeGLX meta (t0 + n / rate)")]
    if k == "openephys":
        return _openephys_streams(src)
    if k == "intan":
        return _intan_streams(src)
    if k == "nwb":
        return _nwb_streams(src)
    if k == "si_folder":
        def _open():
            import spikeinterface.core as sc

            return sc.load(src.path)
        return [Stream(src.name, _open, {"geometry_source": "SpikeInterface folder"}, "SpikeInterface folder times")]
    if k == "binary":
        return [Stream(src.name, lambda: _open_binary(src), {"geometry_source": "probeinterface JSON" if src.probe_json else "none"},
                       "sidecar t_start + n / rate")]
    raise ValueError(f"unknown source kind {k}")


# ---------------------------------------------------------------------------------------------- SpikeGLX

def _spikeglx_streams(src: Source) -> list[Stream]:
    import spikeinterface.extractors as se

    names, ids = se.get_neo_streams("spikeglx", src.path)
    out = []
    for sid in ids:
        if not sid.endswith(".ap"):
            continue
        meta_file = next((f for f in src.files if f".{sid}." in f.name or f.name.endswith(f"{sid}.bin")), None)
        lf = sid[:-3] + ".lf"
        out.append(Stream(sid, (lambda s=sid: se.read_spikeglx(src.path, stream_id=s)),
                          _spikeglx_probe_meta(meta_file) if meta_file else {}, "SpikeGLX meta (t0 + n / rate)",
                          lfp=(lambda s=lf: (se.read_spikeglx(src.path, stream_id=s), None, s)) if lf in ids else None))
    return out


def _spikeglx_probe_meta(bin_file: Path) -> dict:
    meta_file = bin_file.with_name(bin_file.name.rsplit(".", 1)[0] + ".meta")
    out = {"geometry_source": "SpikeGLX meta (IMRO table)"}
    try:
        for line in meta_file.read_text(errors="replace").splitlines():
            key, _, val = line.partition("=")
            if key in ("imDatPrb_pn", "imDatPrb_type"):
                out.setdefault("part_number", val)
            if key == "imDatPrb_sn":
                out["serial"] = val
    except OSError:
        pass
    if "part_number" in out:
        out["model"] = _np_model_name(out["part_number"])
    return out


def _open_cbin(src: Source):
    import spikeinterface.extractors as se

    return se.read_cbin_ibl(src.path, cbin_file_path=src.files[0])


# ---------------------------------------------------------------------------------------------- Open Ephys

def _openephys_streams(src: Source) -> list[Stream]:
    import spikeinterface.extractors as se

    names, _ = se.get_neo_streams("openephys" + ("binary" if _is_binary_oe(src.path) else "legacy"), src.path)
    n_blocks = 1
    try:
        n_blocks = se.get_neo_num_blocks("openephys" + ("binary" if _is_binary_oe(src.path) else "legacy"), src.path)
    except Exception:   # noqa: BLE001
        pass
    out = []
    for name in names:
        low = name.lower()
        if any(t in low for t in ("-lfp", "-lf", "adc", "nidaq", "ni-daq", "sync", "aux", "message", "event")):
            continue
        for b in range(n_blocks):
            label = name if n_blocks == 1 else f"{name}#block{b}"
            lf = next((x for x in names if x != name and x.replace("-LFP", "").replace("-LF", "") == name.replace("-AP", "")), None)
            out.append(Stream(label, (lambda n=name, b=b: se.read_openephys(src.path, stream_name=n, block_index=b)),
                              {"geometry_source": "Open Ephys settings.xml"}, "Open Ephys sample numbers / rate",
                              lfp=(lambda n=lf, b=b: (se.read_openephys(src.path, stream_name=n, block_index=b), None, n)) if lf else None))
    return out


def _is_binary_oe(folder: Path) -> bool:
    for p in folder.rglob("structure.oebin"):
        return True
    return False


# ---------------------------------------------------------------------------------------------- Intan

def _intan_streams(src: Source) -> list[Stream]:
    import spikeinterface.extractors as se

    names, _ = se.get_neo_streams("intan", src.path)
    out = []
    for name in names:
        if "amplifier" not in name.lower():
            continue
        out.append(Stream(name, (lambda n=name: se.read_intan(src.path, stream_name=n, ignore_integrity_checks=True)),
                          {"geometry_source": "probeinterface JSON" if src.probe_json else "none"}, "Intan sample numbers / rate"))
    return out


# ---------------------------------------------------------------------------------------------- NWB

def nwb_identity(path: Path) -> dict:
    import h5py

    out: dict = {}
    with h5py.File(path, "r") as f:
        for key in ("session_start_time", "identifier", "session_description", "timestamps_reference_time"):
            if key in f:
                v = f[key][()]
                out[key] = v.decode() if isinstance(v, bytes) else str(v)
        if "general/subject" in f:
            subj = {}
            g = f["general/subject"]
            for k in g.keys():
                try:
                    v = g[k][()]
                    subj[k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
                except Exception:   # noqa: BLE001
                    pass
            out["subject"] = subj
    return out


def nwb_electrical_series(path: Path, lfp: bool = False) -> list[dict]:
    """ElectricalSeries in acquisition/ (spike band: rate >= 10 kHz; or with lfp=True the LFP ones), with time base and device."""
    import h5py

    found = []
    with h5py.File(path, "r") as f:
        if "acquisition" not in f:
            return []

        def visit(name, obj):
            if not isinstance(obj, h5py.Group):
                return
            ndt = obj.attrs.get("neurodata_type")
            ndt = ndt.decode() if isinstance(ndt, bytes) else ndt
            if ndt != "ElectricalSeries" or "data" not in obj:
                return
            if "starting_time" in obj:
                rate = float(obj["starting_time"].attrs["rate"])
                has_ts = False
            elif "timestamps" in obj:
                ts = obj["timestamps"][:1001]
                rate = float(1.0 / np.median(np.diff(ts))) if len(ts) > 1 else 0.0
                has_ts = True
            else:
                return
            found.append({"path": f"acquisition/{name}", "name": name.split("/")[-1], "rate": rate,
                          "timestamps": has_ts, "shape": tuple(obj["data"].shape)})

        f["acquisition"].visititems(visit)
        for s in found:
            s["device"] = _nwb_device_for(f, s["path"])
    is_lfp = [s["rate"] < LFP_MAX_RATE or bool(re.search(r"(^|[^A-Za-z])LF(P)?($|[^A-Za-z])|LF$|LFP", s["name"])) for s in found]
    return [s for s, l in zip(found, is_lfp) if l == lfp]


def _probe_token(name: str) -> str:
    """The part of a stream name that identifies the probe (Probe01, imec0, ProbeA), lower case; '' when none."""
    m = re.search(r"(probe[_ -]?\w?\d*|imec\d+)", name, re.I)
    return m.group(1).lower() if m else ""


def _nwb_device_for(f, series_path: str) -> dict:
    """The probe model/serial from the electrodes' group device (description, manufacturer, model attributes)."""
    out: dict = {}
    try:
        es = f[series_path]
        region = es["electrodes"]
        table = f[region.attrs["table"]] if "table" in region.attrs else f["general/extracellular_ephys/electrodes"]
        idx = np.asarray(region[:])
        groups = table["group"][idx[:1]] if "group" in table else []
        if len(groups):
            g = f[groups[0]]
            dev = g["device"] if "device" in g else None
            out["electrode_group"] = g.name.split("/")[-1]
            texts = [str(g.attrs.get("description", ""))]
            if dev is not None:
                for k in ("description", "manufacturer", "model_name", "model_number", "serial_number"):
                    v = dev.attrs.get(k)
                    if v is not None:
                        out[k] = v.decode() if isinstance(v, bytes) else str(v)
                        texts.append(out[k])
                for k in ("model_name", "model_number", "serial_number"):
                    if k in dev:
                        v = dev[k][()]
                        out[k] = v.decode() if isinstance(v, bytes) else str(v)
                        texts.append(out[k])
                out["device"] = dev.name.split("/")[-1]
            m = re.search(r"(PRB_\d_\d_\d{4}_\d(_[A-Z])?|NP\d{4}|PRB2_\d_\d{4}_\d|NP\s?1\.0|NP\s?2\.0|Neuropixels\s?[12]\.0)", " ".join(texts))
            if m:
                out["part_number"] = m.group(1)
    except Exception as exc:   # noqa: BLE001 - device metadata is optional
        log.debug("no device info for %s: %s", series_path, exc)
    return out


def _nwb_streams(src: Source) -> list[Stream]:
    series = nwb_electrical_series(src.path)
    if not series:
        log.info("%s: no spike-band ElectricalSeries in acquisition/ (LFP-only or processed file), skipped", src.path.name)
        return []
    ident = nwb_identity(src.path)
    lfp_series = nwb_electrical_series(src.path, lfp=True)
    out = []
    for s in series:
        def _open(p=s["path"]):
            import spikeinterface.extractors as se

            # Without load_time_vector: SpikeInterface 0.105.4 keeps the timestamps as an h5py dataset without a rate,
            # which breaks slicing and can't be rebuilt in worker processes. They are read separately (Stream.timestamps).
            return se.read_nwb_recording(src.path, electrical_series_path=p, load_time_vector=False)
        dev = s.get("device", {})
        probe = {"geometry_source": "NWB electrodes rel_x/rel_y", "device": dev.get("device"),
                 "part_number": dev.get("part_number"), "serial": dev.get("serial_number"),
                 "description": dev.get("description")}
        if probe["part_number"]:
            probe["model"] = _np_model_name(probe["part_number"])
        lfp_match = [l for l in lfp_series if _probe_token(l["name"]) == _probe_token(s["name"])] if len(lfp_series) > 1 else lfp_series
        out.append(Stream(s["name"], _open, {k: v for k, v in probe.items() if v}, "NWB timestamps" if s["timestamps"] else "NWB starting_time + n / rate", ident,
                          (lambda p=s["path"]: read_nwb_timestamps(src.path, p)) if s["timestamps"] else None,
                          (lambda l=lfp_match[0]: _open_nwb_lfp(src.path, l)) if len(lfp_match) == 1 else None))
    return out


# ---------------------------------------------------------------------------------------------- binary fallback

def _open_binary(src: Source):
    import spikeinterface.extractors as se

    meta = json.loads(src.sidecar.read_text())
    gain = meta.get("gain_to_uV")
    offset = meta.get("offset_to_uV", 0.0 if gain is not None else None)
    rec = se.read_binary(src.path, sampling_frequency=float(meta["sampling_frequency"]), dtype=meta.get("dtype", "int16"),
                         num_channels=int(meta["num_channels"]), t_starts=[float(meta.get("t_start", 0.0))],
                         time_axis=int(meta.get("time_axis", 0)), file_offset=int(meta.get("file_offset", 0)),
                         gain_to_uV=gain, offset_to_uV=offset, channel_ids=meta.get("channel_ids"))
    if "brain_area" in meta and len(meta["brain_area"]) == rec.get_num_channels():
        rec.set_property("brain_area", np.asarray(meta["brain_area"]).astype(str))
    return rec


# ---------------------------------------------------------------------------------------------- probes

def _np_model_name(part: str) -> str:
    p = part.upper().replace(" ", "")
    if p.startswith("PRB_1_") or p in ("NP1000", "NP1010", "NP1020", "NP1030", "NP1100", "NP1120", "NP1121", "NP1122", "NP1123", "NP1300", "0", "1.0", "NP1.0", "NEUROPIXELS1.0"):
        return "Neuropixels 1.0"
    if p.startswith("PRB2_") or p.startswith("NP20") or p in ("21", "24", "NP2.0", "NEUROPIXELS2.0"):
        return "Neuropixels 2.0"
    return part


def attach_probe(rec, src: Source, probe_info: dict):
    """Give the recording a probe: its own, else a probeinterface JSON next to it, else a linear 20 um column."""
    import probeinterface as pi

    if rec.has_probe():
        return rec, probe_info
    if src.probe_json is not None:
        pg = pi.read_probeinterface(src.probe_json)
        probe = pg.probes[0] if len(pg.probes) == 1 else pg
        if hasattr(probe, "device_channel_indices") and probe.device_channel_indices is None:
            probe.set_device_channel_indices(np.arange(probe.get_contact_count()))
        if len(pg.probes) > 1:
            rec.set_probegroup(pg, group_mode="by_probe", in_place=True)
        else:
            rec.set_probe(probe, group_mode="auto", in_place=True)
        probe_info = dict(probe_info, geometry_source=f"probeinterface JSON ({src.probe_json.name})")
        model = pg.probes[0].model_name or pg.probes[0].annotations.get("model_name")
        if model:
            probe_info.setdefault("model", model)
        return rec, probe_info
    if "location" in rec.get_property_keys():
        rec.set_dummy_probe_from_locations(rec.get_channel_locations())
        return rec, probe_info
    n = rec.get_num_channels()
    log.warning("%s: no probe geometry (add a probeinterface JSON next to the recording); assuming one linear column, 20 um pitch", src.name)
    locs = np.column_stack([np.zeros(n), np.arange(n) * 20.0])
    rec.set_dummy_probe_from_locations(locs)
    return rec, dict(probe_info, geometry_source="none: linear 20 um column assumed", model=probe_info.get("model", "unknown"))


def neuropixels_sample_shifts(part_number: str | None, n_channels: int) -> np.ndarray | None:
    """ADC inter-sample shifts of a Neuropixels probe recorded in readout-channel order (NWB/binary sources).

    SpikeGLX / Open Ephys readers set these themselves; NWB files don't carry them. From the probe
    part number's ADC sampling table (probeinterface's ProbeTable copy): shift = sample order / cycles.
    """
    if not part_number:
        return None
    try:
        from probeinterface.neuropixels_tools import build_neuropixels_probe

        pn = part_number
        if pn.upper().replace(" ", "") in ("NP1.0", "NEUROPIXELS1.0"):
            pn = "PRB_1_4_0480_1_C"
        probe = build_neuropixels_probe(pn)
    except Exception:   # noqa: BLE001
        return None
    ann = probe.annotations
    table = ann.get("adc_sampling_table")
    ap, lf = ann.get("ap_sample_frequency_hz"), ann.get("lf_sample_frequency_hz")
    n_readout = int(ann.get("num_readout_channels", 0) or 0)
    if not table or ap is None or lf is None or n_channels != n_readout:
        return None
    head, *groups = table.split(")(")
    n_adcs, per_adc = map(int, head.lstrip("(").split(","))
    order = np.zeros(n_readout)
    for slot, g in enumerate(groups):   # each group: the channels the ADCs sample together at one time slot
        chans = np.array(g.replace("(", "").replace(")", "").split(), dtype=int)
        order[chans[chans < n_readout]] = slot
    cycles = int(per_adc * (1 + (lf or 0) / ap))
    return order / cycles



def read_nwb_timestamps(path: Path, series_path: str) -> np.ndarray:
    """The ElectricalSeries' explicit timestamps (seconds, session clock), float64 (8 bytes per sample)."""
    import h5py

    with h5py.File(path, "r") as f:
        ts = f[series_path]["timestamps"]
        out = np.asarray(ts[:], dtype="float64")
        unit = ts.attrs.get("unit", "seconds")
        interval = float(ts.attrs.get("interval", 1))
    if (unit.decode() if isinstance(unit, bytes) else unit) != "seconds" or interval != 1:
        log.warning("%s: timestamps unit %r interval %s (expected seconds, 1)", series_path, unit, interval)
    return out


def _open_nwb_lfp(path: Path, series: dict):
    import spikeinterface.extractors as se

    rec = se.read_nwb_recording(path, electrical_series_path=series["path"], load_time_vector=False)
    return rec, (read_nwb_timestamps(path, series["path"]) if series["timestamps"] else None), series["name"]

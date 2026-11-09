"""
Synthetic extracellular recording with known units, written as ephys-prep inputs (numpy + json only, so the
platform's review generator image can run it; --nwb also writes NWB variants and needs pynwb).

  <out>/input/sub-fixture_ses-01/sub-fixture_ses-01_ecephys.bin    int16, 64 channels, 30 kHz, 20 s
  <out>/input/sub-fixture_ses-01/sub-fixture_ses-01_ecephys.json   sidecar: rate, channels, dtype, gain, t_start=100 s
  <out>/input/sub-fixture_ses-01/sub-fixture_ses-01_ecephys.probe.json   probeinterface: 2 columns x 32 rows, 20 um pitch
  <out>/ground_truth/sub-fixture_ses-01_gt.npz                       spike samples / unit ids / unit depths

Planted: 12 units with biphasic waveforms (80-220 uV), a slow 15 um downward drift, common-mode noise
(slow + 60 Hz), channel 5 dead (flat), channel 40 noisy (8x high-frequency noise).

With --nwb: <out>/nwb/sub-fixnwb_ses-01_ecephys.nwb (one ElectricalSeries with explicit timestamps at a
clock running 0.2 % fast, starting at 1000 s), <out>/nwb2/twoprobes.nwb (two ElectricalSeries + an LFP one) and
<out>/nwb3/sub-badts_ses-01_ecephys.nwb (timestamps that jump to an apparent 60 kHz for the last 5 s, like IBL's).

    python -m tests.make_fixture OUT [--seconds 20] [--nwb]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FS = 30_000.0
GAIN_UV = 2.34375        # Neuropixels 1.0 AP gain 500: 1.2 V / 1024 / 500
N_ROWS, N_COLS, Y_PITCH, X_PITCH = 32, 2, 20.0, 32.0
DEAD, NOISY = 5, 40
T_START = 100.0


def probe_geometry() -> np.ndarray:
    xs = np.repeat(np.arange(N_COLS) * X_PITCH, N_ROWS)
    ys = np.tile(np.arange(N_ROWS) * Y_PITCH, N_COLS)
    order = np.lexsort((xs, ys))              # channel order: by depth, then column (like Neuropixels)
    return np.column_stack([xs[order], ys[order]])


def probe_json(locs: np.ndarray, model: str = "synthetic 2x32 (20 um pitch)") -> dict:
    n = len(locs)
    return {"specification": "probeinterface", "version": "0.4.2", "probes": [{
        "ndim": 2, "si_units": "um", "annotations": {"model_name": model, "manufacturer": "synthetic"},
        "contact_annotations": {}, "contact_positions": locs.tolist(),
        "contact_plane_axes": [[[1.0, 0.0], [0.0, 1.0]]] * n, "contact_shapes": ["square"] * n,
        "contact_shape_params": [{"width": 12}] * n,
        "probe_planar_contour": [[-25.0, float(locs[:, 1].max() + 25)], [-25.0, -25.0], [float(locs[:, 0].max() + 25), -25.0],
                                 [float(locs[:, 0].max() + 25), float(locs[:, 1].max() + 25)]],
        "device_channel_indices": list(range(n)), "contact_ids": [f"e{i}" for i in range(n)]}]}


def waveform(n_before: int, n_after: int, width: float) -> np.ndarray:
    t = np.arange(-n_before, n_after) / FS * 1000      # ms
    trough = -np.exp(-0.5 * (t / (0.12 * width)) ** 2)
    peak = 0.35 * np.exp(-0.5 * ((t - 0.45 * width) / (0.3 * width)) ** 2)
    return trough + peak


def synth(seconds: float = 20.0, n_units: int = 12, seed: int = 0, drift_um: float = 15.0):
    """Returns (int16 traces (samples x channels), locations, ground truth dict)."""
    rng = np.random.default_rng(seed)
    locs = probe_geometry()
    n_ch, n = len(locs), int(seconds * FS)
    x = rng.normal(0, 9.0, size=(n, n_ch)).astype(np.float32)               # thermal / biological noise, uV
    t = np.arange(n) / FS
    common = 25 * np.sin(2 * np.pi * 3.0 * t) + 12 * np.sin(2 * np.pi * 60.0 * t) + rng.normal(0, 4.0, n).cumsum() * 0.02
    x += common[:, None].astype(np.float32)
    x[:, NOISY] += rng.normal(0, 70.0, n).astype(np.float32)
    nb, na = 30, 60
    unit_y = np.linspace(40, locs[:, 1].max() - 30, n_units) + rng.uniform(-10, 10, n_units)
    unit_x = rng.uniform(0, X_PITCH, n_units)
    amps = rng.uniform(80, 220, n_units)
    widths = rng.uniform(0.8, 1.4, n_units)
    rates = rng.uniform(3, 12, n_units)
    spikes, labels = [], []
    for u in range(n_units):
        isi = rng.exponential(1.0 / rates[u], int(rates[u] * seconds * 2) + 10) + 0.002     # 2 ms refractory
        st = np.cumsum(isi)
        st = st[(st > 0.01) & (st < seconds - 0.01)]
        s = (st * FS).astype(np.int64)
        wf = waveform(nb, na, widths[u])
        for k in s:
            y = unit_y[u] - drift_um * (k / n)                                    # the probe drifts: units move down
            d2 = (locs[:, 0] - unit_x[u]) ** 2 + (locs[:, 1] - y) ** 2
            spatial = amps[u] / (1 + d2 / 30.0 ** 2) ** 1.5
            x[k - nb:k + na] += np.outer(wf, spatial).astype(np.float32)
        spikes.append(s)
        labels.append(np.full(len(s), u))
    x[:, DEAD] = rng.normal(0, 0.5, n)                                            # a dead channel
    traces = np.clip(np.round(x / GAIN_UV), -32768, 32767).astype(np.int16)
    sp = np.concatenate(spikes)
    lab = np.concatenate(labels)
    o = np.argsort(sp, kind="stable")
    gt = {"sample_index": sp[o], "unit_index": lab[o], "unit_y_um": unit_y, "unit_x_um": unit_x, "amplitude_uV": amps,
          "firing_rate_hz": rates, "t_start": T_START, "sampling_frequency": FS, "dead_channel": DEAD, "noisy_channel": NOISY,
          "drift_um": drift_um}
    return traces, locs, gt


def write_binary_input(folder: Path, name: str, traces: np.ndarray, locs: np.ndarray, t_start: float = T_START) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    binf = folder / f"{name}.bin"
    traces.tofile(binf)
    (folder / f"{name}.json").write_text(json.dumps({
        "sampling_frequency": FS, "num_channels": int(traces.shape[1]), "dtype": "int16", "gain_to_uV": GAIN_UV,
        "offset_to_uV": 0.0, "t_start": t_start, "time_axis": 0}, indent=2))
    (folder / f"{name}.probe.json").write_text(json.dumps(probe_json(locs), indent=1))
    return binf


def write_nwb(path: Path, series: list[dict], locs: np.ndarray, identifier: str = "fixture-nwb"):
    """series: [{"name", "data" (samples x channels int16), "timestamps" or "starting_time", "rate", "lfp": bool}]"""
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject

    nwb = NWBFile(session_description="synthetic recording", identifier=identifier,
                  session_start_time=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
                  subject=Subject(subject_id="fix01", species="Mus musculus", sex="F", age="P90D"))
    nwb.add_electrode_column("rel_x", "x on the probe (um)")
    nwb.add_electrode_column("rel_y", "y on the probe (um)")
    regions = ["CA1", "DG", "VISp", "PO"]
    starts = {}
    for s in series:
        probe = s.get("probe", "Probe00")
        if probe in starts:
            continue
        dev = nwb.create_device(name=probe, description="Neuropixels 1.0 (synthetic)" if s.get("np") else "synthetic", manufacturer="synthetic")
        grp = nwb.create_electrode_group(name=probe, description="probe", location="brain", device=dev)
        starts[probe] = len(nwb.electrodes) if nwb.electrodes is not None else 0
        for i, (x, y) in enumerate(locs):
            nwb.add_electrode(group=grp, location=regions[min(3, int(y // 160))], rel_x=float(x), rel_y=float(y))
    for s in series:
        first = starts[s.get("probe", "Probe00")]
        reg = nwb.create_electrode_table_region(list(range(first, first + s["data"].shape[1])), "electrodes")
        kw = {"timestamps": s["timestamps"]} if "timestamps" in s else {"starting_time": s.get("starting_time", 0.0), "rate": s["rate"]}
        nwb.add_acquisition(ElectricalSeries(name=s["name"], data=s["data"], electrodes=reg, conversion=GAIN_UV * 1e-6, **kw))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def write(out: Path, seconds: float = 20.0, nwb: bool = False) -> dict:
    out = Path(out)
    traces, locs, gt = synth(seconds)
    name = "sub-fixture_ses-01_ecephys"
    write_binary_input(out / "input" / "sub-fixture_ses-01", name, traces, locs)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    np.savez(out / "ground_truth" / "sub-fixture_ses-01_gt.npz", **gt)
    if nwb:
        n = traces.shape[0]
        ts = 1000.0 + np.arange(n) / (FS * 1.002)          # explicit timestamps; the clock runs 0.2 % fast
        write_nwb(out / "nwb" / "sub-fixnwb_ses-01_ecephys.nwb", [{"name": "ElectricalSeriesAP", "data": traces, "timestamps": ts, "np": True}], locs)
        bad = 1000.0 + np.arange(n) / FS                     # broken sync: the last 5 s squeezed into 2.5 s (apparent 60 kHz)
        cut = n - int(5 * FS)
        bad[cut:] = bad[cut] + np.arange(n - cut) / (2 * FS)
        write_nwb(out / "nwb3" / "sub-badts_ses-01_ecephys.nwb", [{"name": "ElectricalSeriesAP", "data": traces, "timestamps": bad}], locs,
                  identifier="fixture-bad-timestamps")
        k = int(6 * FS)
        lfp = traces[: int(6 * 2500)].copy()
        write_nwb(out / "nwb2" / "twoprobes.nwb", [
            {"name": "ElectricalSeriesProbe00AP", "data": traces[:k], "starting_time": 5.0, "rate": FS, "probe": "Probe00"},
            {"name": "ElectricalSeriesProbe01AP", "data": traces[:k, ::-1].copy(), "starting_time": 5.0, "rate": FS, "probe": "Probe01"},
            {"name": "ElectricalSeriesProbe00LF", "data": lfp, "starting_time": 5.0, "rate": 2500.0, "probe": "Probe00"},
        ], locs, identifier="fixture-two-probes")
    return gt


def cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--nwb", action="store_true", help="also write NWB variants (needs pynwb)")
    a = ap.parse_args()
    write(a.out, a.seconds, a.nwb)
    print(f"fixture written to {a.out}")


if __name__ == "__main__":
    cli()

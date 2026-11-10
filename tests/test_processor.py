"""ephys-prep without Docker: the synthetic fixture (binary + sidecar, NWB with timestamps, two probes, broken timestamps)."""
from __future__ import annotations

import csv
import os
import json
from pathlib import Path

import numpy as np
import pytest

from processor import inputs as I
from processor import main as M
from processor import preprocess as P
from processor import readers as R
from tests import make_fixture


def run(monkeypatch, input_dir: Path, output_dir: Path, **params) -> int:
    for k in M.PARAMETERS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("THREADS", "2")
    for k, v in params.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixture")
    make_fixture.write(out, seconds=12.0, nwb=True)
    return out


@pytest.fixture(scope="session")
def default_run(fixture_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out-default")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture_dir, out) == 0
    finally:
        mp.undo()
    return out / "recording" / "ephys-prep"


def load(folder):
    import spikeinterface.core as sc

    return sc.load(folder)


def test_bad_parameters(monkeypatch, tmp_path):
    assert run(monkeypatch, tmp_path, tmp_path / "o", MOTION_CORRECTION="sometimes") == 2
    assert run(monkeypatch, tmp_path, tmp_path / "o", FREQ_MIN="7000", FREQ_MAX="6000") == 2
    assert run(monkeypatch, tmp_path, tmp_path / "o", DURATION="-1") == 2


def test_no_input(monkeypatch, tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("nothing here")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o") == 1


def test_discovery_and_session_ids(fixture_dir):
    sources, sheet = I.find_sources(fixture_dir)
    kinds = sorted((s.kind, s.path.name) for s in sources)
    assert ("binary", "sub-fixture_ses-01_ecephys.bin") in kinds
    assert sum(k == "nwb" for k, _ in kinds) == 3
    ids = {I.session_id_for(s, fixture_dir) for s in sources}
    assert {"sub-fixture_ses-01", "sub-fixnwb_ses-01", "twoprobes", "sub-badts_ses-01"} <= ids
    two = next(s for s in sources if s.path.name == "twoprobes.nwb")
    labels = [st.label for st in R.list_streams(two)]
    assert labels == ["ElectricalSeriesProbe00AP", "ElectricalSeriesProbe01AP"]      # the LF series is skipped
    assert I.probe_suffix("imec1.ap", 0) == "_probe01" and I.probe_suffix("Neuropix-PXI-100.ProbeB-AP", 0) == "_probe01"


def test_uuid_session_id(tmp_path):
    p = tmp_path / "sub-NR-0031" / "sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614_desc-raw_ecephys.nwb"
    src = I.Source("nwb", p, p.stem)
    assert I.session_id_for(src, tmp_path) == "sub-NR-0031_ses-caa5dddc"


def test_outputs(default_run):
    rows = list(csv.DictReader(open(default_run / "sessions.csv")))
    ids = {r["session_id"]: r for r in rows}
    assert set(ids) == {"sub-fixture_ses-01", "sub-fixnwb_ses-01", "twoprobes_probe00", "twoprobes_probe01", "sub-badts_ses-01"}
    assert all(r["status"] == "ok" for r in rows)
    sdir = default_run / "sub-fixture_ses-01"
    for f in ("recording/si_folder.json", "motion", "session.json", "sub-fixture_ses-01_raw_traces.png", "sub-fixture_ses-01_bad_channels.png",
              "sub-fixture_ses-01_drift_raster.png", "sub-fixture_ses-01_motion.png"):
        assert (sdir / f).exists(), f
    s = json.loads((sdir / "session.json").read_text())
    for k in ("session_id", "source", "stream", "probe", "sampling_frequency", "t_start", "t_stop", "duration_s", "n_channels",
              "bad_channels", "motion_corrected", "motion_method", "time_source"):
        assert k in s, k
    assert s["motion_corrected"] is False and s["motion_estimated"] is True and s["motion_above_threshold"] is True   # 15 um planted
    assert "spike-sort's CPU sorters interpolate" in s["motion_decision"]
    prov = json.loads((default_run / "provenance.json").read_text())
    assert set(prov) >= {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished", "wall_seconds",
                         "peak_rss_mb", "gpu", "sessions", "citations"}
    summ = json.loads((default_run / "summary.json").read_text())
    qc = default_run.parent.parent / "qc" / "ephys-prep"                 # the small-files port
    assert (qc / "summary.json").exists() and (qc / "sub-fixture_ses-01" / "session.json").exists()
    assert not (qc / "sub-fixture_ses-01" / "recording").exists() and not (qc / "sub-fixture_ses-01" / "motion").exists()
    assert all((qc / f["path"]).exists() for f in summ["figures"]) and all(t["path"] and t["caption"] for t in summ["tables"])
    assert all(f["caption"] for f in summ["figures"]) and summ["sessions"]["sub-fixture_ses-01"]["n_bad_channels"] >= 2


def test_bad_channels_found(default_run):
    s = json.loads((default_run / "sub-fixture_ses-01" / "session.json").read_text())
    bad = {b["id"] for b in s["bad_channels"]}
    assert {str(make_fixture.DEAD), str(make_fixture.NOISY)} <= bad and len(bad) <= 4      # the planted dead and noisy channels
    reasons = {b["id"]: b["reason"] for b in s["bad_channels"]}
    assert reasons[str(make_fixture.DEAD)].startswith("dead") and reasons[str(make_fixture.NOISY)].startswith("noisy")
    rec = load(default_run / "sub-fixture_ses-01" / "recording")
    assert rec.get_num_channels() == 64 - len(bad) and str(make_fixture.DEAD) not in {str(c) for c in rec.channel_ids}


def test_session_clock_kept(default_run):
    rec = load(default_run / "sub-fixture_ses-01" / "recording")
    assert abs(rec.get_start_time() - make_fixture.T_START) < 1e-9          # sidecar t_start = 100 s, not re-zeroed
    nwb = load(default_run / "sub-fixnwb_ses-01" / "recording")
    assert nwb.has_time_vector()
    t = nwb.get_times()
    assert abs(t[0] - 1000.0) < 1e-9 and abs((t[1] - t[0]) - 1 / (make_fixture.FS * 1.002)) < 1e-12   # the file's own timestamps
    s = json.loads((default_run / "sub-fixnwb_ses-01" / "session.json").read_text())
    assert s["time_source"] == "NWB timestamps" and abs(s["clock_difference_percent"] - 0.2) < 0.01
    assert s["session_start_time"].startswith("2026-01-02") and s["identifier"] == "fixture-nwb" and s["subject"]["subject_id"] == "fix01"
    assert s["institution"] == "Fixture University" and s["lab"] == "fixture lab" and s["nwb_session_id"] == "fix-session-1"
    assert s["experimenter"] == ["Doe, Jane"] and s["subject"]["sex"] == "F" and s["subject"]["age"] == "P90D"
    assert set(nwb.get_property("brain_area")) <= {"CA1", "DG", "VISp", "PO"}


def test_broken_timestamps_dropped(default_run):
    s = json.loads((default_run / "sub-badts_ses-01" / "session.json").read_text())
    tp = s["timestamp_problems"]
    assert tp["action"] == "drop" and len(tp["bad_stretches"]) == 1
    rec = load(default_run / "sub-badts_ses-01" / "recording")
    t = rec.get_times()
    assert t[-1] < 1000.0 + 7.0 + 1e-6 and np.all(np.diff(t) > 0)      # 12 s fixture, the last 5 s squeezed: 7 s kept
    summ = json.loads((default_run / "summary.json").read_text())
    assert "timestamp_problems" in summ["sessions"]["sub-badts_ses-01"]


def test_window_on_session_clock(monkeypatch, fixture_dir, tmp_path):
    assert run(monkeypatch, fixture_dir / "input", tmp_path, T_START="103.5", DURATION="4", MOTION_CORRECTION="none") == 0
    rec = load(tmp_path / "recording" / "ephys-prep" / "sub-fixture_ses-01" / "recording")
    assert abs(rec.get_start_time() - 103.5) < 1 / make_fixture.FS and abs(rec.get_total_duration() - 4.0) < 1e-3
    assert not (tmp_path / "recording" / "ephys-prep" / "sub-fixture_ses-01" / "motion").exists()


def test_window_outside_fails_session(monkeypatch, fixture_dir, tmp_path):
    assert run(monkeypatch, fixture_dir / "input", tmp_path, T_START="5000") == 1


def test_motion_apply_and_stream_selection(monkeypatch, fixture_dir, tmp_path):
    assert run(monkeypatch, fixture_dir / "nwb2", tmp_path, MOTION_CORRECTION="apply", MOTION_THRESHOLD_UM="0", STREAM="Probe01",
               BAD_CHANNELS="none") == 0
    ids = [r["session_id"] for r in csv.DictReader(open(tmp_path / "recording" / "ephys-prep" / "sessions.csv"))]
    assert ids == ["twoprobes_probe01"]          # the suffix stays: the file has two probes
    s = json.loads((tmp_path / "recording" / "ephys-prep" / "twoprobes_probe01" / "session.json").read_text())
    assert s["motion_corrected"] is True and s["n_channels"] <= 64 and abs(s["t_start"] - 5.0) < 1e-9
    assert s["motion_decision"].startswith("interpolated here")


def test_motion_threshold_and_lfp(monkeypatch, fixture_dir, tmp_path):
    """Drift below MOTION_THRESHOLD_UM is not interpolated even with apply; MOTION_SOURCE=lfp uses the probe's LF series."""
    assert run(monkeypatch, fixture_dir / "nwb2", tmp_path, MOTION_CORRECTION="apply", MOTION_THRESHOLD_UM="1000", STREAM="Probe00",
               MOTION_SOURCE="lfp") == 0
    s = json.loads((tmp_path / "recording" / "ephys-prep" / "twoprobes_probe00" / "session.json").read_text())
    assert s["motion_corrected"] is False and s["motion_above_threshold"] is False and s["motion_estimated"] is True
    assert s["motion_source"] == "lfp" and s["motion"]["estimate_params"]["method"] == "dredge_lfp"
    assert s["motion"]["estimate_params"]["lfp_stream"] == "ElectricalSeriesProbe00LF"
    m = load_motion(tmp_path / "recording" / "ephys-prep" / "twoprobes_probe00" / "motion")
    assert 5.0 <= m.temporal_bins_s[0][0] < 11.0                    # session clock


def load_motion(folder):
    from spikeinterface.core import Motion

    return Motion.load(folder)


def test_preprocessed_signal(default_run, fixture_dir):
    """Spikes survive, common-mode noise is gone: the planted units' troughs stand out of the noise."""
    gt = np.load(fixture_dir / "ground_truth" / "sub-fixture_ses-01_gt.npz")
    rec = load(default_run / "sub-fixture_ses-01" / "recording")
    x = rec.get_traces(start_frame=0, end_frame=rec.get_num_samples(), return_in_uV=True)
    noise = np.median(np.abs(x), axis=0) / 0.6745
    assert np.median(noise) < 12                        # 9 uV white noise; the 25 uV common sine is removed
    peaks = [x[s - 2:s + 3].min() for s in gt["sample_index"][:200] if s > 10]
    assert np.median(peaks) < -6 * np.median(noise)


def test_check_timestamps_unit():
    fs = 1000.0
    t = np.arange(20_000) / fs
    r = P.check_timestamps(t, fs)
    assert r["ok"] and not r["pauses"] and abs(r["effective_rate"] - fs) < 1e-6
    # IBL-like sync fault: the last quarter runs at twice the rate -> dropped, the longest good stretch kept
    t2 = t.copy()
    t2[15_000:] = t2[15_000] + np.arange(5000) / (2 * fs)
    r = P.check_timestamps(t2, fs)
    assert not r["ok"] and r["keep"] == (0, 15_000)
    # a 10-sample glitch (stuck clock) is repaired, nothing dropped
    t3 = t.copy()
    t3[3000:3010] = t3[2999]
    r = P.check_timestamps(t3, fs)
    assert r["ok"] and r["keep"] == (0, 20_000) and r["repaired"] and np.allclose(r["times"], t)
    # 10 samples jumping backwards by 1 s: repaired too
    t4 = t.copy()
    t4[8000:8010] -= 1.0
    r = P.check_timestamps(t4, fs)
    assert r["ok"] and r["keep"] == (0, 20_000) and np.allclose(r["times"], t)
    # a 30 s acquisition pause: kept, recorded, not dropped; the effective rate ignores the gap
    t5 = t.copy()
    t5[12_000:] += 30.0
    r = P.check_timestamps(t5, fs)
    assert r["ok"] and r["keep"] == (0, 20_000) and r["pauses"] == [(12_000, pytest.approx(30.001))]
    assert abs(r["effective_rate"] - fs) < 1e-6


def test_np1_sample_shifts():
    s = R.neuropixels_sample_shifts("PRB_1_4_0480_1_C", 384)
    assert s is not None and s[0] == 0 and s[1] == 0 and abs(s[2] - 1 / 13) < 1e-12 and abs(s.max() - 11 / 13) < 1e-12
    assert R.neuropixels_sample_shifts("PRB_1_4_0480_1_C", 64) is None


@pytest.mark.skipif(not os.environ.get("EPHYS_FORMAT_DATA"), reason="set EPHYS_FORMAT_DATA to a folder with spikeglx/, openephys/, intan/ samples")
@pytest.mark.parametrize("fmt,kind", [("spikeglx", "spikeglx"), ("openephys", "openephys"), ("intan", "intan")])
def test_vendor_formats(monkeypatch, tmp_path, fmt, kind):
    """SpikeGLX / Open Ephys binary / Intan readers on real files (e.g. from the NEO ephy_testing_data on GIN,
    see sample-input/README.md); not in the repository."""
    src = Path(os.environ["EPHYS_FORMAT_DATA"]) / fmt
    assert run(monkeypatch, src, tmp_path) == 0
    rows = list(csv.DictReader(open(tmp_path / "recording" / "ephys-prep" / "sessions.csv")))
    assert rows and all(r["status"] == "ok" and r["source_format"] == kind for r in rows)


def test_sheet_columns_do_not_overwrite_measurements(monkeypatch, fixture_dir, tmp_path):
    """A sessions.csv with columns named like measured values (t_start, probe) must not replace them in summary.json."""
    import shutil

    shutil.copytree(fixture_dir / "input", tmp_path / "in")
    (tmp_path / "in" / "sessions.csv").write_text("session_id,path,subject_id,t_start,probe\nS1,sub-fixture_ses-01,fix01,999,Probe01\n")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", MOTION_CORRECTION="none") == 0
    h = json.loads((tmp_path / "out" / "recording" / "ephys-prep" / "summary.json").read_text())["sessions"]["S1"]
    assert h["t_start"] == pytest.approx(100.0) and h["subject_id"] == "fix01" and h["probe"] != "Probe01"


def test_pause_kept_and_recorded_time(monkeypatch, tmp_path):
    """A 12 s acquisition pause in explicit timestamps: kept (no data dropped); duration_s is the span on the
    session clock, recorded_s the time actually recorded, which rates use."""
    traces, locs, _ = make_fixture.synth(10.0)
    n = traces.shape[0]
    ts = 500.0 + np.arange(n) / make_fixture.FS
    ts[n // 2:] += 12.0
    make_fixture.write_nwb(tmp_path / "in" / "sub-pause_ses-01_ecephys.nwb", [{"name": "ElectricalSeriesAP", "data": traces, "timestamps": ts}],
                           locs, identifier="fixture-pause")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", MOTION_CORRECTION="none") == 0
    top = tmp_path / "out" / "recording" / "ephys-prep"
    s = json.loads((top / "sub-pause_ses-01" / "session.json").read_text())
    assert len(s["timestamp_problems"]["pauses"]) == 1 and "bad_stretches" not in s["timestamp_problems"]
    assert s["duration_s"] == pytest.approx(22.0, abs=1e-3) and s["recorded_s"] == pytest.approx(10.0, abs=1e-3)
    assert s["sampling_frequency"] == pytest.approx(make_fixture.FS, rel=1e-9)
    rec = load(top / "sub-pause_ses-01" / "recording")
    assert rec.get_num_samples() == n and rec.get_times()[n // 2] - rec.get_times()[n // 2 - 1] == pytest.approx(12.0 + 1 / make_fixture.FS)
    h = json.loads((top / "summary.json").read_text())["sessions"]["sub-pause_ses-01"]
    assert h["recorded_s"] == pytest.approx(10.0, abs=1e-3)

"""ephys-prep without Docker: the synthetic fixture (binary + sidecar, NWB with timestamps, two probes, broken timestamps)."""
from __future__ import annotations

import csv
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
    return out / "ephys-prep"


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
    assert s["motion_corrected"] is False and s["motion_estimated"] is True
    prov = json.loads((default_run / "provenance.json").read_text())
    assert set(prov) >= {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished", "wall_seconds",
                         "peak_rss_mb", "gpu", "sessions", "citations"}
    summ = json.loads((default_run / "summary.json").read_text())
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
    rec = load(tmp_path / "ephys-prep" / "sub-fixture_ses-01" / "recording")
    assert abs(rec.get_start_time() - 103.5) < 1 / make_fixture.FS and abs(rec.get_total_duration() - 4.0) < 1e-3
    assert not (tmp_path / "ephys-prep" / "sub-fixture_ses-01" / "motion").exists()


def test_window_outside_fails_session(monkeypatch, fixture_dir, tmp_path):
    assert run(monkeypatch, fixture_dir / "input", tmp_path, T_START="5000") == 1


def test_motion_apply_and_stream_selection(monkeypatch, fixture_dir, tmp_path):
    assert run(monkeypatch, fixture_dir / "nwb2", tmp_path, MOTION_CORRECTION="apply", STREAM="Probe01", BAD_CHANNELS="none") == 0
    ids = [r["session_id"] for r in csv.DictReader(open(tmp_path / "ephys-prep" / "sessions.csv"))]
    assert ids == ["twoprobes_probe01"]          # the suffix stays: the file has two probes
    s = json.loads((tmp_path / "ephys-prep" / "twoprobes_probe01" / "session.json").read_text())
    assert s["motion_corrected"] is True and s["n_channels"] <= 64 and abs(s["t_start"] - 5.0) < 1e-9


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
    assert P.check_timestamps(t, fs)["ok"]
    t2 = t.copy()
    t2[15_000:] = t2[15_000] + np.arange(5000) / (2 * fs)
    r = P.check_timestamps(t2, fs)
    assert not r["ok"] and r["keep"] == (0, 15_000)
    t3 = t.copy()
    t3[3000:3010] = t3[2999]
    r = P.check_timestamps(t3, fs)
    assert not r["ok"] and r["keep"][0] >= 3000


def test_np1_sample_shifts():
    s = R.neuropixels_sample_shifts("PRB_1_4_0480_1_C", 384)
    assert s is not None and s[0] == 0 and s[1] == 0 and abs(s[2] - 1 / 13) < 1e-12 and abs(s.max() - 11 / 13) < 1e-12
    assert R.neuropixels_sample_shifts("PRB_1_4_0480_1_C", 64) is None

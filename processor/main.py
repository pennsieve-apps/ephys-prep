"""
ephys-prep: raw extracellular recordings -> preprocessed recordings ready for spike sorting.

Reads SpikeGLX, Open Ephys, Intan, NWB ElectricalSeries (incl. IBL raw NWB with timestamps),
SpikeInterface folders and raw binary files with a JSON sidecar, anywhere under INPUT_DIR. Every
probe of every recording is a session. Per session it writes, under OUTPUT_DIR/ephys-prep/<session_id>/:

  recording/      SpikeInterface binary_folder, int16, probe + channel locations (+ brain_area), session-clock times
  motion/         SpikeInterface Motion (DREDge) when estimated
  session.json    what the next steps need (conventions.md)
  <session_id>_raw_traces.png, _bad_channels.png, _drift_raster.png, _motion.png

and ephys-prep/sessions.csv, summary.json, provenance.json.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from processor import common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("ephys-prep")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py", "hdmf", "pynwb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP = "ephys-prep"
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "T_START": "",
    "DURATION": "0",
    "STREAM": "",
    "FILTER": "highpass",
    "FREQ_MIN": "300",
    "FREQ_MAX": "6000",
    "PHASE_SHIFT": "auto",
    "BAD_CHANNELS": "auto",
    "EXCLUDE_CHANNELS": "",
    "MAX_BAD_CHANNEL_FRACTION": "0.5",
    "REFERENCE": "cmr",
    "MOTION_CORRECTION": "estimate",
    "MOTION_PRESET": "dredge_fast",
    "MOTION_SOURCE": "ap",
    "MOTION_THRESHOLD_UM": "10",
    "BAD_TIMESTAMPS": "drop",
    "THREADS": "0",
    "SEED": "42",
}
PARAMETERS = tuple(DEFAULTS)
CHOICES = {
    "FILTER": ("highpass", "bandpass"),
    "PHASE_SHIFT": ("auto", "true", "false"),
    "BAD_CHANNELS": ("auto", "dead-noisy", "none"),
    "REFERENCE": ("cmr", "destripe", "none"),
    "MOTION_CORRECTION": ("estimate", "apply", "none"),
    "MOTION_PRESET": ("dredge_fast", "dredge"),
    "MOTION_SOURCE": ("ap", "lfp"),
    "BAD_TIMESTAMPS": ("drop", "keep"),
}
CITATIONS = {
    "spikeinterface": "10.7554/eLife.61834",
    "aind_pipeline": "10.7554/eLife.110170",
    "dredge": "10.1038/s41592-025-02614-5",
    "drift_benchmark": "10.1523/ENEURO.0229-23.2023",
    "ibl_destripe": "10.6084/m9.figshare.19705522",
}
SESSION_COLUMNS = ["session_id", "path", "subject_id", "condition", "group", "status", "source_format", "stream", "probe_model",
                   "t_start", "t_stop", "duration_s", "n_channels_source", "n_channels", "n_bad_channels", "motion_corrected",
                   "max_drift_range_um", "n_peaks", "seconds", "error"]


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    for k, choices in CHOICES.items():
        v = raw[k].strip().lower()
        if v not in choices:
            raise ParameterError(f"{k} must be one of {', '.join(choices)}; got {raw[k]!r}")
        p[k] = v
    try:
        p["T_START"] = float(raw["T_START"]) if raw["T_START"].strip() else None
        p["DURATION"] = float(raw["DURATION"])
        p["FREQ_MIN"] = float(raw["FREQ_MIN"])
        p["FREQ_MAX"] = float(raw["FREQ_MAX"])
        p["MAX_BAD_CHANNEL_FRACTION"] = float(raw["MAX_BAD_CHANNEL_FRACTION"])
        p["MOTION_THRESHOLD_UM"] = float(raw["MOTION_THRESHOLD_UM"])
        p["THREADS"] = int(raw["THREADS"])
        p["SEED"] = int(raw["SEED"])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if p["DURATION"] < 0 or p["THREADS"] < 0 or p["MOTION_THRESHOLD_UM"] < 0:
        raise ParameterError("DURATION, THREADS and MOTION_THRESHOLD_UM must be >= 0")
    if not 0 < p["FREQ_MIN"] < p["FREQ_MAX"]:
        raise ParameterError("need 0 < FREQ_MIN < FREQ_MAX")
    if not 0 < p["MAX_BAD_CHANNEL_FRACTION"] <= 1:
        raise ParameterError("MAX_BAD_CHANNEL_FRACTION must be in (0, 1]")
    p["STREAM"] = [s.strip() for s in raw["STREAM"].split(",") if s.strip()]
    p["EXCLUDE_CHANNELS"] = [s.strip() for s in raw["EXCLUDE_CHANNELS"].split(",") if s.strip()]
    return p


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("spikeinterface", "probeinterface", "neo", "pynwb", "hdmf", "h5py", "numpy", "scipy", "numba", "matplotlib", "mtscomp"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------- sessions

def plan_sessions(sources, sheet_rows, input_dir: Path, p: dict) -> list[dict]:
    """Expand sources into sessions (one per selected probe stream and recording segment)."""
    from processor import inputs as I
    from processor import readers as R

    planned, used = [], set()
    for src in sources:
        try:
            streams = R.list_streams(src)
        except Exception as exc:   # noqa: BLE001 - one unreadable source must not stop the others
            log.error("%s: cannot list streams (%s: %s)", _rel(src.path, input_dir), type(exc).__name__, exc)
            planned.append({"session_id": I.SAFE_ID.sub("-", src.name), "source": src, "stream": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        multi = len(streams) > 1
        row = I.match_sheet(src, sheet_rows, input_dir)
        base = row["session_id"] if row else I.session_id_for(src, input_dir)
        selected = [s for s in streams if not p["STREAM"] or any(x.lower() in s.label.lower() for x in p["STREAM"])]
        if streams and not selected:
            log.info("%s: no stream matches STREAM=%s (streams: %s)", _rel(src.path, input_dir), ",".join(p["STREAM"]),
                     ", ".join(s.label for s in streams))
        for i, st in enumerate(streams):
            if st not in selected:
                continue
            sid = base + (I.probe_suffix(st.label, i) if multi else "")
            planned.append({"session_id": sid, "source": src, "stream": st, "sheet": row or {}})
    # unique ids
    for s in planned:
        sid, k = s["session_id"], 2
        while sid in used:
            sid = f"{s['session_id']}-{k}"
            k += 1
        if sid != s["session_id"]:
            log.warning("session id %s is used twice; the second is %s", s["session_id"], sid)
        s["session_id"] = sid
        used.add(sid)
    return planned


def process_session(item: dict, p: dict, out_root: Path, input_dir: Path, job_kwargs: dict, warnings: list) -> list[dict]:
    """One planned session (may expand into one session per recording segment). Returns result rows."""
    import numpy as np

    from processor import figures as F
    from processor import preprocess as P
    from processor import readers as R

    src, st = item["source"], item["stream"]
    rec_all = st.open()
    rec_all, probe_info = R.attach_probe(rec_all, src, dict(st.probe))
    if str(probe_info.get("geometry_source", "")).startswith("none"):
        msg = (f"{item['session_id']}: NO PROBE GEOMETRY for {src.path.name} ({rec_all.get_num_channels()} channels): one 20 um column "
               "was assumed, so bad-channel detection and drift are unreliable and drift is not estimated. Add a probeinterface "
               "JSON next to the recording (<name>.probe.json).")
        log.warning(msg)
        warnings.append(msg)
    n_seg = rec_all.get_num_segments()
    results = []
    for seg in range(n_seg):
        sid = item["session_id"] + (f"_rec{seg:02d}" if n_seg > 1 else "")
        t0 = time.monotonic()
        row = {"session_id": sid, "path": _rel(src.path, input_dir), "source_format": src.kind, "stream": st.label,
               "probe_model": probe_info.get("model", ""), **{k: v for k, v in item.get("sheet", {}).items() if k not in ("session_id", "path")}}
        try:
            rec = rec_all.select_segments([seg]) if n_seg > 1 else rec_all
            info = _process_one(sid, rec, src, st, probe_info, item, p, out_root, input_dir, job_kwargs, warnings, P, F, R, np)
            row.update(info)
            row["status"] = "ok"
        except Exception as exc:   # noqa: BLE001 - a failed session is logged and skipped
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log.error("session %s failed: %s", sid, row["error"])
            if not isinstance(exc, P.SessionError):
                log.debug("traceback", exc_info=True)
            shutil.rmtree(out_root / sid, ignore_errors=True)
        row["seconds"] = round(time.monotonic() - t0, 1)
        results.append(row)
    return results


def _process_one(sid, rec, src, st, probe_info, item, p, out_root, input_dir, job_kwargs, warnings, P, F, R, np) -> dict:
    sdir = out_root / sid
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    fs = float(rec.get_sampling_frequency())
    n_src = rec.get_num_channels()

    # time base: explicit timestamps vs nominal rate; broken stretches of timestamps
    time_info = {"time_source": st.time_source}
    tv = st.timestamps() if st.timestamps is not None else None
    if tv is not None and len(tv) != rec.get_num_samples():
        raise P.SessionError(f"{len(tv)} timestamps for {rec.get_num_samples()} samples")
    t_first, t_last = (float(tv[0]), float(tv[-1]) + 1.0 / fs) if tv is not None else (float(rec.get_start_time()), float(rec.get_end_time()) + 1.0 / fs)
    log.info("session %s: %s %s, %d channels at %.1f Hz (%s), %.1f s on the session clock [%.3f, %.3f)", sid, src.kind, st.label, n_src, fs,
             st.time_source, t_last - t_first, t_first, t_last)
    if tv is not None:
        chk = P.check_timestamps(np.asarray(tv), fs)
        tv = chk["times"]
        tp: dict = {}
        if chk["pauses"]:
            tp["pauses"] = [{"sample": i, "t": float(tv[i]), "gap_s": g} for i, g in chk["pauses"]]
            msg = (f"{sid}: {len(chk['pauses'])} pause(s) in the recording ("
                   + "; ".join(f"{g:.3f} s at {float(tv[i]):.3f} s" for i, g in chk["pauses"][:5])
                   + "): kept; the timestamps carry the gaps (sorters see the samples back to back)")
            log.warning(msg)
            warnings.append(msg)
        if chk["repaired"]:
            tp["repaired"] = [{"from_sample": a, "to_sample": b, "t": float(tv[a])} for a, b in chk["repaired"]]
            msg = (f"{sid}: {len(chk['repaired'])} timestamp glitch(es) (backwards / repeated / jumping samples) repaired by "
                   "interpolation between their good neighbours: " + "; ".join(f"samples {a}-{b}" for a, b in chk["repaired"][:5]))
            log.warning(msg)
            warnings.append(msg)
        if chk["bad_blocks"] or not chk["ok"]:
            bad = [[float(tv[a]), float(tv[min(b, len(tv) - 1)]), int(b - a)] for a, b in chk["bad_blocks"]]
            k0, k1 = chk["keep"]
            msg = (f"{sid}: timestamps are broken in {len(bad)} stretch(es) "
                   + "; ".join(f"{a:.3f}-{b:.3f} s ({n} samples)" for a, b, n in bad[:5])
                   + f" (a local rate more than 0.1 % off the median {chk['median_rate']:.3f} Hz, or an unrepairable glitch)")
            if p["BAD_TIMESTAMPS"] == "drop":
                msg += f"; BAD_TIMESTAMPS=drop keeps the longest good stretch, samples {k0}-{k1} ({float(tv[k0]):.3f}-{float(tv[k1 - 1]):.3f} s)"
                rec = rec.frame_slice(k0, k1)
                tv = tv[k0:k1]
            else:
                msg += "; BAD_TIMESTAMPS=keep: the timestamps are used as they are"
            log.warning(msg)
            warnings.append(msg)
            tp.update(bad_stretches=[{"t_from": a, "t_to": b, "n_samples": n} for a, b, n in bad], action=p["BAD_TIMESTAMPS"],
                      kept_samples=[k0, k1] if p["BAD_TIMESTAMPS"] == "drop" else None)
        if tp:
            tp["median_rate"] = chk["median_rate"]
            time_info["timestamp_problems"] = tp
        if len(tv) > 1:
            eff = float(chk["effective_rate"] or (len(tv) - 1) / float(tv[-1] - tv[0]))
            nominal = nominal_rate(eff, probe_info)
            drift = (eff - nominal) / nominal * 100
            time_info.update(effective_sampling_frequency=eff, nominal_sampling_frequency=nominal, clock_difference_percent=drift,
                             sampling_frequency_source="effective rate of the timestamps (pauses excluded); the times file is authoritative")
            if abs(drift) > 0.1:
                msg = (f"{sid}: timestamps give an effective rate of {eff:.3f} Hz vs {nominal:.3f} Hz nominal ({drift:+.3f} %); "
                       "times come from the timestamps")
                log.warning(msg)
                warnings.append(msg)
            # The recording's sampling_frequency: the effective rate of the timestamps, not SpikeInterface's estimate from
            # the first 1000 intervals (29,997.0 Hz for IBL vs 30,000.48 Hz over the session: 70 ms off after 10 min).
            R.set_sampling_frequency(rec, eff)
            fs = eff

    shifts = None
    if "inter_sample_shift" not in rec.get_property_keys():
        shifts = R.neuropixels_sample_shifts(probe_info.get("part_number") or probe_info.get("model"), n_src)
    prep = P.prepare(rec, p, shifts, job_kwargs, times=tv)
    del tv
    log.info("session %s: window [%.3f, %.3f) s, %d bad channels (%s), phase shift %s, %s reference [peak RSS %s MB]", sid, prep.t_start, prep.t_stop,
             len(prep.bad_channels), ", ".join(b["id"] for b in prep.bad_channels[:12]) + (" ..." if len(prep.bad_channels) > 12 else ""),
             "yes" if prep.phase_shift else "no", p["REFERENCE"], C.peak_rss_mb())

    figures = []
    cap = F.bad_channels(sdir / f"{sid}_bad_channels.png", sid, prep.filtered, prep.channel_labels, prep.noise_uV)
    bad_fig = {"path": f"{sid}/{sid}_bad_channels.png", "caption": cap}

    final = prep.preprocessed
    motion = motion_info = motion_session = None
    motion_method = "none"
    motion_stats: dict = {}
    motion_decision = "motion not estimated (MOTION_CORRECTION=none)"
    above = None
    applied = False
    n_peaks = None
    if p["MOTION_CORRECTION"] != "none":
        ok, why = P.motion_feasible(prep.for_motion)
        if str(probe_info.get("geometry_source", "")).startswith("none"):
            ok, why = False, "no probe geometry (a 20 um column was assumed)"
        if not ok:
            log.info("session %s: motion not estimated: %s", sid, why)
            motion_method = f"none ({why})"
            motion_decision = f"motion not estimated: {why}"
        else:
            t_m = time.monotonic()
            source = p["MOTION_SOURCE"]
            if source == "lfp":
                lfp = st.lfp() if st.lfp is not None else None
                if lfp is None and "Neuropixels 1.0" in str(probe_info.get("model", "")):
                    msg = f"{sid}: MOTION_SOURCE=lfp but this recording has no LFP stream (Neuropixels 1.0 AP data are high-passed in hardware); using spikes (dredge_ap)"
                    log.warning(msg)
                    warnings.append(msg)
                    source = "ap"
            if source == "lfp":
                with C.Heartbeat(f"{sid}: LFP motion estimation (DREDge LFP)"):
                    if lfp is not None:
                        lrec, ltimes, lname = lfp
                        lwin, _, _, lwt = P.apply_window(lrec, prep.t_start, prep.t_stop - prep.t_start, ltimes)
                        lwin = __import__("spikeinterface.preprocessing", fromlist=["x"]).scale_to_uV(lwin) if lwin.has_scaleable_traces() else lwin
                    else:                       # a wide-band recording: the LFP recipe low-passes and resamples it
                        lname, lwin, lwt = f"{st.label} (wide band)", prep.raw_uV, prep.times
                    lm, mparams = P.estimate_motion_lfp(lwin, [b["id"] for b in prep.bad_channels], job_kwargs)
                    motion_session = P.motion_to_session_clock(lm, lwin, lwt)
                    motion = P.motion_to_rec_clock(motion_session, prep.for_motion, prep.times)
                    mparams["lfp_stream"] = lname
                motion_label = "DREDge LFP"
            else:
                with C.Heartbeat(f"{sid}: motion estimation (detect, localise, DREDge)"):
                    motion, motion_info, mparams = P.estimate_motion(prep.for_motion, p, job_kwargs)
                n_peaks = int(len(motion_info["peaks"]))
                motion_session = P.motion_to_session_clock(motion, prep.for_motion, prep.times)
                motion_label = f"DREDge ({p['MOTION_PRESET']})"
            motion_session.save(sdir / "motion")
            cap, motion_stats = F.motion_traces(sdir / f"{sid}_motion.png", sid, motion_session)
            motion_fig = {"path": f"{sid}/{sid}_motion.png", "caption": cap}
            above = motion_stats["max_drift_range_um"] > p["MOTION_THRESHOLD_UM"]
            applied = p["MOTION_CORRECTION"] == "apply" and above
            size = f"{motion_stats['max_drift_range_um']:.1f} um {'>' if above else '<='} MOTION_THRESHOLD_UM={p['MOTION_THRESHOLD_UM']:g}"
            if p["MOTION_CORRECTION"] == "apply":
                motion_decision = (f"interpolated here ({size})" if applied else f"not interpolated: drift too small to be worth interpolating ({size})")
            else:
                motion_decision = ("estimated only; " + (f"drift is above the threshold ({size}): spike-sort's CPU sorters interpolate with "
                                                         "this motion, Kilosort4 corrects drift itself" if above else
                                                         f"drift is small ({size}): spike-sort's CPU sorters do not correct it; Kilosort4 still runs its own correction"))
            motion_method = f"{motion_label}, " + ("estimated and applied (kriging interpolation)" if applied else "estimated only")
            log.info("session %s: motion (%s) estimated in %.0f s%s; drift range median %.1f um, max %.1f um; %s [peak RSS %s MB]", sid, motion_label, time.monotonic() - t_m,
                     f" from {n_peaks} peaks" if n_peaks else "", motion_stats["median_drift_range_um"], motion_stats["max_drift_range_um"], motion_decision,
                     C.peak_rss_mb())
            if applied:
                from spikeinterface.sortingcomponents.motion import interpolate_motion

                final = interpolate_motion(prep.for_motion, motion, border_mode="force_extrapolate", spatial_interpolation_method="kriging",
                                           sigma_um=20.0, p=2, dtype="float32")
            motion_stats["estimate_params"] = mparams

    gain = P.choose_int16_gain(prep, final)
    out = P.to_int16(final, gain)
    t_s = time.monotonic()
    with C.Heartbeat(f"{sid}: writing the preprocessed recording"):
        out.save(folder=sdir / "recording", format="binary", overwrite=True, **job_kwargs)
    if prep.times is not None:
        P.attach_times_file(sdir / "recording", prep.times)
    log.info("session %s: wrote %d channels x %d samples (int16, %.4g uV/bit) in %.0f s [peak RSS %s MB]", sid, out.get_num_channels(), out.get_num_samples(), gain,
             time.monotonic() - t_s, C.peak_rss_mb())

    # figures (read back the saved data for the 'after' panel: it is what the sorter sees)
    import spikeinterface.core as sc

    saved = sc.load(sdir / "recording")
    saved_uV = __import__("spikeinterface.preprocessing", fromlist=["scale_to_uV"]).scale_to_uV(saved)
    to_time = P.session_time(prep.times, prep.raw_uV)
    cap = F.raw_vs_preprocessed(sdir / f"{sid}_raw_traces.png", sid, prep.raw_uV, saved_uV, prep.channel_labels,
                                prep.raw_uV.get_num_samples() // 2, to_time)
    figures.append({"path": f"{sid}/{sid}_raw_traces.png", "caption": cap})
    if motion_info is not None:
        cap = F.drift_raster(sdir / f"{sid}_drift_raster.png", sid, prep.for_motion, motion_info["peaks"], motion_info["peak_locations"], motion,
                             applied=applied, to_time=to_time)
        figures.append({"path": f"{sid}/{sid}_drift_raster.png", "caption": cap})
    if motion is not None:
        figures.append(motion_fig)
    figures.append(bad_fig)

    sheet = {k: v for k, v in item.get("sheet", {}).items() if k not in ("session_id", "path")}
    nwb = st.nwb or {}
    session = {
        "session_id": sid,
        "source": _rel(src.path, input_dir),
        "source_format": src.kind,
        "stream": st.label,
        "probe": {k: probe_info.get(k) for k in ("model", "serial", "part_number", "geometry_source") if probe_info.get(k)},
        "sampling_frequency": float(out.get_sampling_frequency()),
        "t_start": prep.t_start,
        "t_stop": prep.t_stop,
        "duration_s": prep.t_stop - prep.t_start,
        "n_channels": out.get_num_channels(),
        "n_channels_source": n_src,
        "bad_channels": prep.bad_channels,
        "motion_corrected": applied,
        "motion_estimated": motion is not None,
        "motion_method": motion_method,
        "motion_source": p["MOTION_SOURCE"] if motion is not None else None,
        "motion_threshold_um": p["MOTION_THRESHOLD_UM"],
        "motion_above_threshold": above,
        "motion_decision": motion_decision,
        "motion": {k: v for k, v in motion_stats.items()} if motion_stats else None,
        **time_info,
        "gain_to_uV": gain,
        "dtype": "int16",
        "has_brain_area": "brain_area" in out.get_property_keys(),
        "preprocessing": prep.steps + ([{"step": "interpolate_motion", "border_mode": "force_extrapolate", "method": "kriging"}]
                                       if applied else []),
        "session_start_time": nwb.get("session_start_time"),
        "identifier": nwb.get("identifier"),
        "session_description": nwb.get("session_description"),
        "timestamps_reference_time": nwb.get("timestamps_reference_time"),
        "subject": nwb.get("subject"),
        **{k: nwb.get(k) for k in ("institution", "lab", "experimenter", "experiment_description", "keywords", "nwb_session_id", "protocol")},
        **({"sheet": sheet} if sheet else {}),
        "app": APP,
        "version": app_version(),
    }
    session = {k: v for k, v in session.items() if v is not None}
    C.write_json(sdir / "session.json", session)
    return {"t_start": prep.t_start, "t_stop": prep.t_stop, "duration_s": prep.t_stop - prep.t_start, "n_channels_source": n_src,
            "n_channels": out.get_num_channels(), "n_bad_channels": len(prep.bad_channels), "motion_corrected": session["motion_corrected"],
            "max_drift_range_um": motion_stats.get("max_drift_range_um"), "n_peaks": n_peaks, "_figures": figures, "_session": session,
            "_motion": motion_stats}


STANDARD_RATES = (20_000.0, 24_000.0, 24_414.0625, 25_000.0, 30_000.0, 32_000.0, 40_000.0, 48_000.0)


def nominal_rate(effective: float, probe_info: dict) -> float:
    """The acquisition system's nominal rate: 30 kHz for Neuropixels, else the nearest standard rate within 1 %."""
    if "Neuropixels" in str(probe_info.get("model", "")):
        return 30_000.0
    best = min(STANDARD_RATES, key=lambda r: abs(r - effective))
    return best if abs(best - effective) / best < 0.01 else effective


def session_motion_applied(motion, p) -> bool:
    return motion is not None and p["MOTION_CORRECTION"] == "apply"


# --------------------------------------------------------------------------------------------- main

def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    try:
        p = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    threads = C.set_thread_env(p["THREADS"])
    C.setup_writable_dirs()
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1

    import numpy as np
    import spikeinterface.core as sc

    np.random.seed(p["SEED"])
    # spawn: each worker re-opens the recording from its description. Forked workers would share the parent's
    # HDF5 (NWB) file handle, which HDF5 does not support (workers crash on compressed NWB data), and forking
    # after PyTorch has run (DREDge) is unsafe too.
    job_kwargs = {"n_jobs": C.worker_count(threads), "chunk_duration": "1s", "progress_bar": False, "mp_context": "spawn"}
    sc.set_global_job_kwargs(**job_kwargs)
    log.info("%s %s: THREADS=%d, parameters %s", APP, app_version(), threads,
             {k: v for k, v in p.items() if k not in ("THREADS",)})

    from processor import inputs as I

    sources, sheet = I.find_sources(input_dir)
    sheet_rows = I.read_sheet(sheet)
    if sheet:
        log.info("sessions sheet: %s (%d rows)", _rel(sheet, input_dir), len(sheet_rows))
    log.info("found %d recording(s): %s", len(sources), ", ".join(f"{s.kind}:{_rel(s.path, input_dir)}" for s in sources) or "none")
    if not sources:
        log.error("no recordings found under %s (SpikeGLX *.ap.bin + .meta, Open Ephys, Intan *.rhd/*.rhs, NWB, "
                  "SpikeInterface folders, or <name>.bin + <name>.json sidecar)", input_dir)
        return 1
    planned = plan_sessions(sources, sheet_rows, input_dir, p)
    if not any(x.get("stream") for x in planned):
        log.error("no spike-band streams to process (check STREAM)")
        return 1

    out_root = output_dir / APP
    out_root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    rows: list[dict] = []
    for item in planned:
        if item.get("stream") is None:
            rows.append({"session_id": item["session_id"], "path": _rel(item["source"].path, input_dir), "status": "failed",
                         "source_format": item["source"].kind, "error": item.get("error")})
            continue
        try:
            rows.extend(process_session(item, p, out_root, input_dir, job_kwargs, warnings))
        except Exception as exc:   # noqa: BLE001 - opening failed
            log.error("session %s failed to open: %s: %s", item["session_id"], type(exc).__name__, exc)
            rows.append({"session_id": item["session_id"], "path": _rel(item["source"].path, input_dir), "status": "failed",
                         "source_format": item["source"].kind, "stream": item["stream"].label, "error": f"{type(exc).__name__}: {exc}"})
        r = rows[-1]
        log.info("session %s: %s in %s s%s", r["session_id"], r["status"], r.get("seconds", "?"),
                 f", {r.get('n_channels')} channels, {r.get('n_bad_channels')} bad, {r.get('duration_s', 0):.1f} s" if r["status"] == "ok" else
                 f" ({r.get('error')})")

    ok = [r for r in rows if r["status"] == "ok"]
    extra_cols = sorted({k for r in rows for k in r if not k.startswith("_") and k not in SESSION_COLUMNS})
    C.write_csv(out_root / "sessions.csv", SESSION_COLUMNS + extra_cols, rows)

    figures = [f for r in ok for f in r["_figures"]]
    summary = {
        "app": APP, "version": app_version(),
        "sessions": {r["session_id"]: _headline(r) for r in rows},
        "figures": figures,
        "tables": [{"path": "sessions.csv", "caption": "One row per session: source, window, channels, bad channels, drift."}],
        "warnings": warnings + [f"{r['session_id']}: failed ({r.get('error')})" for r in rows if r["status"] != "ok"],
    }
    C.write_json(out_root / "summary.json", summary)
    cites = [CITATIONS["spikeinterface"], CITATIONS["aind_pipeline"]]
    if any(r.get("_session", {}).get("motion_estimated") for r in ok):
        cites += [CITATIONS["dredge"], CITATIONS["drift_benchmark"]]
    if p["REFERENCE"] == "destripe":
        cites.append(CITATIONS["ibl_destripe"])
    finished = datetime.now(timezone.utc)
    C.write_json(out_root / "provenance.json", {
        "app": APP, "version": app_version(), "variant": None,
        "tools": tool_versions(),
        "parameters": {k: (",".join(v) if isinstance(v, list) else v) for k, v in p.items()},
        "inputs": sorted({_rel(f, input_dir) for s in sources for f in (s.files or [s.path])}),
        "started": started.isoformat().replace("+00:00", "Z"), "finished": finished.isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t0, 1),
        "peak_rss_mb": C.peak_rss_mb(),
        "gpu": None,
        "sessions": {r["session_id"]: {"status": r["status"], "seconds": r.get("seconds"), **({"error": r["error"]} if r.get("error") else {})}
                     for r in rows},
        "citations": cites,
    })
    log.info("%s done: %d of %d sessions ok in %.0f s, peak RSS %s MB", APP, len(ok), len(rows), time.monotonic() - t0, C.peak_rss_mb())
    return 0 if ok else 1


def _headline(r: dict) -> dict:
    if r["status"] != "ok":
        return {"status": r["status"], "error": r.get("error")}
    s = r["_session"]
    m = r.get("_motion") or {}
    out = {"status": "ok", "source_format": s["source_format"], "stream": s["stream"], "probe": s.get("probe", {}).get("model"),
           "duration_s": round(s["duration_s"], 3), "t_start": s["t_start"], "t_stop": s["t_stop"],
           "sampling_frequency": s["sampling_frequency"], "n_channels": s["n_channels"], "n_bad_channels": len(s["bad_channels"]),
           "motion_estimated": s["motion_estimated"], "motion_corrected": s["motion_corrected"],
           "motion_above_threshold": s.get("motion_above_threshold"), "motion_decision": s.get("motion_decision"),
           "median_drift_range_um": m.get("median_drift_range_um"), "max_drift_range_um": m.get("max_drift_range_um"),
           "n_peaks": r.get("n_peaks"), "peaks_per_s": (r["n_peaks"] / s["duration_s"]) if r.get("n_peaks") else None,
           "time_source": s["time_source"]}
    if "timestamp_problems" in s:
        out["timestamp_problems"] = s["timestamp_problems"]
    if "clock_difference_percent" in s:
        out["effective_sampling_frequency"] = s["effective_sampling_frequency"]
        out["clock_difference_percent"] = s["clock_difference_percent"]
    out.update({k: v for k, v in s.get("sheet", {}).items() if k not in out})    # sheet columns never overwrite measured values
    return out

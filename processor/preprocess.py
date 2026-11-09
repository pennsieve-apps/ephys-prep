"""
The preprocessing of one session (one probe), after the AIND ephys pipeline v1.5.0 (Buccino et al. 2026),
which is also IBL's recipe:

  time window -> (phase shift) -> high-pass 300 Hz (or band-pass) -> bad channels (coherence+psd) removed
  -> common median reference (or IBL destriping) -> DREDge motion estimation (and optionally interpolation)
  -> int16 binary folder with the session-clock times.

Everything is lazy (SpikeInterface chains); only the final save and the motion step read the data, in
1 s chunks over THREADS workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from processor.common import log

MIN_CHANNELS_FOR_COHERENCE = 32      # coherence+psd needs neighbours along the shank
MIN_CHANNELS_FOR_MOTION = 16
MIN_SPAN_FOR_MOTION_UM = 150.0
AIND_DESTRIPE = dict(n_channel_pad=60, n_channel_taper=None, direction="y", apply_agc=True, agc_window_length_s=0.01,
                     highpass_butter_order=3, highpass_butter_wn=0.01)
AIND_BAD_CHANNELS = dict(method="coherence+psd", dead_channel_threshold=-0.5, noisy_channel_threshold=1.0,
                         outside_channel_threshold=-0.3, outside_channels_location="top", n_neighbors=11,
                         channel_filters=["noise", "dead", "out"])


class SessionError(RuntimeError):
    """A session cannot be processed (logged, the session is skipped)."""


@dataclass
class Prepared:
    raw_uV: object                   # the windowed source recording in uV (float32), for the QC figure
    filtered: object                 # after phase shift + filter, all channels (bad-channel figure)
    preprocessed: object             # float32 uV, bad channels removed, re-referenced, (motion-corrected)
    for_motion: object               # what motion is estimated on (preprocessed, before interpolation)
    t_start: float
    t_stop: float
    steps: list = field(default_factory=list)
    bad_channels: list = field(default_factory=list)   # [{"id", "reason"}]
    channel_labels: dict = field(default_factory=dict)  # id -> label (good / dead / noise / out / excluded)
    noise_uV: dict = field(default_factory=dict)       # id -> noise (MAD) after filtering
    phase_shift: bool = False
    source_gain_uV: float | None = None
    times: object = None             # the window's timestamps when the source has explicit ones (re-attached before saving)


def apply_window(rec, t_start: float | None, duration: float, times: np.ndarray | None = None):
    """Slice [T_START, T_START + DURATION) on the session clock (times are kept, never re-zeroed).

    `times`: explicit sample times (NWB timestamps), else the recording's t_start + n / rate.
    Returns (recording, t_start, t_stop, times of the window or None).
    """
    fs = float(rec.get_sampling_frequency())
    n = rec.get_num_samples()
    if times is not None:
        t0, t1 = float(times[0]), float(times[-1]) + 1.0 / fs
    else:
        t0, t1 = float(rec.get_start_time()), _stop(rec)
    if t_start is None and duration <= 0:
        return rec, t0, t1, times
    start = t0 if t_start is None else float(t_start)
    if start >= t1:
        raise SessionError(f"T_START={start:.3f} s is after the recording's end ({t1:.3f} s on the session clock; it starts at {t0:.3f} s)")
    end = t1 if duration <= 0 else min(t1, start + duration)
    if end <= t0:
        raise SessionError(f"the window [{start:.3f}, {end:.3f}) s ends before the recording starts ({t0:.3f} s)")
    if start < t0:
        log.warning("T_START=%.3f s is before the recording starts (%.3f s); the window starts at %.3f s", start, t0, t0)
        start = t0
    if times is not None:
        i0, i1 = int(np.searchsorted(times, start, side="left")), int(np.searchsorted(times, end, side="left"))
    else:
        i0 = int(np.ceil((start - t0) * fs - 1e-6))
        i1 = min(n, int(np.ceil((end - t0) * fs - 1e-6)))
    if i1 - i0 < int(fs):
        raise SessionError(f"the window [{start:.3f}, {end:.3f}) s holds less than 1 s of data")
    sub = rec.frame_slice(i0, i1)
    if times is not None:
        wt = times[i0:i1]
        return sub, float(wt[0]), float(wt[-1]) + 1.0 / fs, wt
    return sub, float(sub.get_start_time()), _stop(sub), None


def _stop(rec) -> float:
    """End of the recording on the session clock: the last sample's time plus one sample period."""
    return float(rec.get_end_time()) + 1.0 / float(rec.get_sampling_frequency())


def prepare(rec, p: dict, sample_shifts: np.ndarray | None, job_kwargs: dict, times: np.ndarray | None = None) -> Prepared:
    import spikeinterface.preprocessing as spre

    rec, t_start, t_stop, times = apply_window(rec, p["T_START"], p["DURATION"], times)
    steps = [{"step": "window", "t_start": t_start, "t_stop": t_stop}]
    source_gain = None
    if rec.has_scaleable_traces():
        gains = np.asarray(rec.get_channel_gains(), dtype=float)
        source_gain = float(np.median(gains)) if np.issubdtype(rec.get_dtype(), np.integer) else None
        rec_uV = spre.scale_to_uV(rec)
    else:
        log.warning("the recording has no gain to uV; values are kept in the file's units and called uV")
        rec_uV = spre.astype(rec, "float32")
    steps.append({"step": "scale_to_uV", "source_dtype": str(rec.get_dtype()), "source_gain_uV": source_gain})

    x = rec_uV
    did_shift = False
    if p["PHASE_SHIFT"] != "false":
        shifts = rec.get_property("inter_sample_shift") if "inter_sample_shift" in rec.get_property_keys() else sample_shifts
        if shifts is not None and np.any(np.asarray(shifts) > 0):
            x = spre.phase_shift(x, margin_ms=100.0, inter_sample_shift=np.asarray(shifts, dtype=float), dtype="float32")
            did_shift = True
            steps.append({"step": "phase_shift", "margin_ms": 100.0, "max_shift_samples": float(np.max(shifts))})
        elif p["PHASE_SHIFT"] == "true":
            log.warning("PHASE_SHIFT=true but the probe has no ADC inter-sample shifts (not a Neuropixels probe the readers know); skipped")

    nyq = rec.get_sampling_frequency() / 2
    if p["FILTER"] == "bandpass":
        fmax = min(p["FREQ_MAX"], 0.95 * nyq)
        x = spre.bandpass_filter(x, freq_min=p["FREQ_MIN"], freq_max=fmax, dtype="float32")
        steps.append({"step": "bandpass_filter", "freq_min": p["FREQ_MIN"], "freq_max": fmax})
    else:
        x = spre.highpass_filter(x, freq_min=p["FREQ_MIN"], dtype="float32")
        steps.append({"step": "highpass_filter", "freq_min": p["FREQ_MIN"]})
    filtered = x

    # Bad channels: detected on the filtered recording, as AIND does.
    from spikeinterface.core import get_noise_levels

    ids = list(filtered.channel_ids)
    noise = get_noise_levels(filtered, return_in_uV=False, random_slices_kwargs={"seed": p["SEED"]}, **job_kwargs)
    noise_uV = {str(c): float(v) for c, v in zip(ids, noise)}
    labels = {str(c): "good" for c in ids}
    bad: list[dict] = []
    if p["BAD_CHANNELS"] in ("auto", "dead-noisy"):
        n = len(ids)
        kw = dict(AIND_BAD_CHANNELS) if n >= MIN_CHANNELS_FOR_COHERENCE else {"method": "mad"}
        if p["BAD_CHANNELS"] == "dead-noisy" and "channel_filters" in kw:
            kw["channel_filters"] = ["noise", "dead"]          # keep channels that look outside the brain
        if n < MIN_CHANNELS_FOR_COHERENCE:
            log.info("%d channels: bad channels by noise level (mad) instead of coherence+psd", n)
        bad_ids, ch_labels = spre.detect_bad_channels(filtered, seed=p["SEED"], **kw)
        for c, lab in zip(ids, ch_labels):
            if c in set(bad_ids):
                labels[str(c)] = str(lab)
                bad.append({"id": str(c), "reason": {"dead": "dead (low coherence with neighbours)", "noise": "noisy (high-frequency power)",
                                                     "out": "outside the brain (top of the probe)"}.get(str(lab), str(lab))})
        steps.append({"step": "detect_bad_channels", **{k: v for k, v in kw.items()}, "n_bad": len(bad)})
    for c in p["EXCLUDE_CHANNELS"]:
        match = [x_ for x_ in ids if str(x_) == c]
        if not match:
            log.warning("EXCLUDE_CHANNELS: no channel %r in this recording", c)
            continue
        if labels[c] == "good":
            labels[c] = "excluded"
            bad.append({"id": c, "reason": "excluded (EXCLUDE_CHANNELS)"})
    frac = len(bad) / max(1, len(ids))
    if frac > p["MAX_BAD_CHANNEL_FRACTION"]:
        raise SessionError(f"{len(bad)} of {len(ids)} channels are bad ({frac:.0%} > MAX_BAD_CHANNEL_FRACTION={p['MAX_BAD_CHANNEL_FRACTION']:.0%}); "
                           "check the recording (or set BAD_CHANNELS=none)")
    if bad:
        bad_set = {b["id"] for b in bad}
        x = x.remove_channels([c for c in ids if str(c) in bad_set])
    if x.get_num_channels() < 1:
        raise SessionError("no channels left after removing bad channels")

    if p["REFERENCE"] == "cmr":
        x = spre.common_reference(x, reference="global", operator="median", dtype="float32")
        steps.append({"step": "common_reference", "reference": "global", "operator": "median"})
    elif p["REFERENCE"] == "destripe":
        x = spre.highpass_spatial_filter(x, dtype="float32", **AIND_DESTRIPE)
        steps.append({"step": "highpass_spatial_filter (IBL destriping)", **AIND_DESTRIPE})

    # The session-clock timestamps (when the source has explicit ones) stay in `times`, not on the recordings:
    # SpikeInterface sends every recording to its worker processes, and a time vector (8 bytes per sample, and not
    # passed on by every preprocessor) would travel with each one. They are applied to motion bins, figures and the
    # saved folder (times_cached_seg0.raw) by the caller.
    return Prepared(raw_uV=rec_uV, filtered=filtered, preprocessed=x, for_motion=x, t_start=t_start, t_stop=t_stop, steps=steps,
                    bad_channels=bad, channel_labels=labels, noise_uV=noise_uV, phase_shift=did_shift, source_gain_uV=source_gain,
                    times=times)


def motion_feasible(rec) -> tuple[bool, str]:
    n = rec.get_num_channels()
    depth = rec.get_channel_locations()[:, 1]
    span = float(depth.max() - depth.min()) if n else 0.0
    if n < MIN_CHANNELS_FOR_MOTION or span < MIN_SPAN_FOR_MOTION_UM:
        return False, f"{n} channels over {span:.0f} um: too few / too short a probe to estimate drift"
    return True, ""


def estimate_motion(rec, p: dict, job_kwargs: dict):
    """DREDge (preset MOTION_PRESET) on the preprocessed recording. Returns (motion, motion_info, params)."""
    import spikeinterface.preprocessing as spre

    depth = rec.get_channel_locations()[:, 1]
    span = float(depth.max() - depth.min())
    # AIND sets the non-rigid window to 10 % of the probe's span (win_step_norm = win_scale_norm = 0.1):
    # 382 um on a Neuropixels 1.0 column of 3,820 um, close to the preset's 400 um. Shorter probes: rigid.
    est = {}
    if span >= 400.0:
        est = {"win_step_um": max(50.0, 0.1 * span), "win_scale_um": max(50.0, 0.1 * span)}
    else:
        est = {"rigid": True}
    motion, info = spre.compute_motion(rec, preset=p["MOTION_PRESET"], estimate_motion_kwargs=est, output_motion_info=True,
                                       raise_error=True, **job_kwargs)
    return motion, info, {"preset": p["MOTION_PRESET"], **est}


def to_int16(rec, gain_uV: float):
    """float32 uV -> int16 with `gain_uV` uV per bit, clipped to the int16 range (rounded, never wrapped)."""
    import spikeinterface.preprocessing as spre

    lim = 32767 * gain_uV
    clipped = spre.clip(rec, a_min=-lim, a_max=lim)
    out = spre.scale(clipped, gain=1.0 / gain_uV, offset=0.0, dtype="int16")
    out.set_channel_gains(gain_uV)
    out.set_channel_offsets(0.0)
    return out


def choose_int16_gain(prep: Prepared, rec_final) -> float:
    """Keep the source's quantisation step when it was an integer recording; else ~1/20 of the median noise."""
    if prep.source_gain_uV and prep.source_gain_uV > 0:
        return float(prep.source_gain_uV)
    med = float(np.median(list(prep.noise_uV.values()))) if prep.noise_uV else 10.0
    return max(med / 20.0, 0.01)


def check_timestamps(times: np.ndarray, fs: float, tol: float = 0.001, block_s: float = 1.0) -> dict:
    """Find stretches where explicit timestamps are broken: not increasing, or a local rate off the median by > tol.

    The rate is measured over blocks of `block_s` (single sample intervals are too noisy). Returns
    {"ok": bool, "median_rate", "bad_blocks": [[i0, i1), ...] in samples, "keep": (i0, i1) the longest good stretch}.
    IBL's raw NWB is an example: after a sync-edge mismatch its timestamps run at an apparent 60 kHz.
    """
    n = len(times)
    B = max(2, int(round(block_s * fs)))
    if n < 2 * B:
        mono = bool(np.all(np.diff(times) > 0))
        return {"ok": mono, "median_rate": None, "bad_blocks": [] if mono else [[0, n]], "keep": (0, n)}
    edges = np.arange(0, n, B)
    if edges[-1] != n - 1:
        edges = np.append(edges, n - 1)
    span = np.diff(times[edges])
    nsamp = np.diff(edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = nsamp / span
    med = float(np.median(rate[np.isfinite(rate) & (rate > 0)]))
    good = np.isfinite(rate) & (rate > 0) & (np.abs(rate - med) / med <= tol)
    # non-monotonic samples anywhere in a block make it bad
    bad_idx = np.flatnonzero(np.diff(times) <= 0)
    if len(bad_idx):
        good[np.clip(np.searchsorted(edges, bad_idx, side="right") - 1, 0, len(good) - 1)] = False
    if good.all():
        return {"ok": True, "median_rate": med, "bad_blocks": [], "keep": (0, n)}
    bad_blocks, best, cur = [], (0, 0), None
    for k, g in enumerate(good):
        if g:
            cur = (cur[0], k + 1) if cur else (k, k + 1)
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
            i0, i1 = int(edges[k]), int(edges[k + 1])
            if bad_blocks and bad_blocks[-1][1] == i0:
                bad_blocks[-1][1] = i1
            else:
                bad_blocks.append([i0, i1])
    keep = (int(edges[best[0]]), int(edges[best[1]]) + (1 if best[1] == len(good) else 0))
    return {"ok": False, "median_rate": med, "bad_blocks": bad_blocks, "keep": keep}


def session_time(prep_times, rec):
    """sample index (of `rec`) -> seconds on the session clock: the explicit timestamps, else t_start + i / rate."""
    if prep_times is not None:
        return lambda idx: np.asarray(prep_times)[np.clip(np.asarray(idx, dtype=np.int64), 0, len(prep_times) - 1)]
    return lambda idx: rec.sample_index_to_time(np.asarray(idx))


def motion_to_session_clock(motion, rec, prep_times):
    """A copy of `motion` whose temporal bins are on the session clock (they come out of DREDge in the recording's
    nominal clock, t_start + i / rate, which drifts from explicit timestamps)."""
    if prep_times is None:
        return motion
    from spikeinterface.core import Motion

    fs, t0 = rec.get_sampling_frequency(), rec.get_start_time()
    bins = []
    for tb in motion.temporal_bins_s:
        idx = (np.asarray(tb) - t0) * fs
        bins.append(np.interp(idx, np.arange(len(prep_times)), prep_times))
    return Motion(displacement=motion.displacement, temporal_bins_s=bins, spatial_bins_um=motion.spatial_bins_um,
                  direction=motion.direction, interpolation_method=motion.interpolation_method)


def attach_times_file(folder, times) -> None:
    """Give a saved binary_folder its explicit session-clock timestamps: times_cached_seg0.raw (float64, one per
    sample) referenced from binary.json, exactly as SpikeInterface writes it for a recording with a time vector."""
    import json
    from pathlib import Path

    folder = Path(folder)
    np.ascontiguousarray(times, dtype="<f8").tofile(folder / "times_cached_seg0.raw")
    for name in ("binary.json", "si_folder.json"):
        f = folder / name
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        kw = d.get("kwargs", {})
        if "file_paths" in kw:                      # binary.json (BinaryRecordingExtractor)
            kw["file_timestamps_paths"] = ["times_cached_seg0.raw"]
            kw["t_starts"] = None
            f.write_text(json.dumps(d, indent=4))


def estimate_motion_lfp(lfp, bad_ids, job_kwargs: dict):
    """DREDge on the LFP (Windolf et al. 2025; SpikeInterface's dredge_lfp recipe): band-pass 0.5-250 Hz, common
    median reference, resample to 250 Hz, second spatial derivative (a current-source-density-like signal),
    average the columns, then rigid registration of 1 s bins. Better than spikes when units are sparse or the
    motion is fast (human intraoperative / NHP recordings). Returns (motion, params)."""
    import spikeinterface.preprocessing as spre
    from spikeinterface.sortingcomponents.motion import estimate_motion

    x = lfp
    drop = [c for c in x.channel_ids if str(c) in set(bad_ids)]
    if drop:
        x = x.remove_channels(drop)
    # DREDge LFP reads the whole (short, 250 Hz) signal at once, so long filter margins avoid chunk-edge artefacts
    x = spre.bandpass_filter(x, freq_min=0.5, freq_max=250.0, margin_ms=3000.0, ignore_low_freq_error=True, dtype="float32")
    x = spre.common_reference(x, reference="global", operator="median")
    x = spre.resample(x, resample_rate=250, margin_ms=1000)
    x = spre.directional_derivative(x, order=2, edge_order=1)
    x = spre.average_across_direction(x)
    motion = estimate_motion(x, method="dredge_lfp", rigid=True, progress_bar=False)
    return motion, {"method": "dredge_lfp", "rigid": True, "bandpass_hz": [0.5, 250.0], "resample_hz": 250}


def lfp_from_ap(rec):
    """An LFP signal from a wide-band recording (low-pass + decimate is done by the LFP recipe's band-pass and resample)."""
    return rec


def motion_to_rec_clock(motion_session, rec, prep_times):
    """Inverse of motion_to_session_clock: temporal bins from the session clock to `rec`'s nominal clock."""
    if prep_times is None:
        return motion_session
    from spikeinterface.core import Motion

    fs, t0 = rec.get_sampling_frequency(), rec.get_start_time()
    bins = [t0 + np.interp(np.asarray(tb), prep_times, np.arange(len(prep_times))) / fs for tb in motion_session.temporal_bins_s]
    return Motion(displacement=motion_session.displacement, temporal_bins_s=bins, spatial_bins_um=motion_session.spatial_bins_um,
                  direction=motion_session.direction, interpolation_method=motion_session.interpolation_method)

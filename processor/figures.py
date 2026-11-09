"""QC figures (PNG, 150 dpi, session id in the title). Each returns the caption the report shows."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DPI = 150
LABEL_COLOURS = {"good": "#4a4a4a", "dead": "#1f77b4", "noise": "#d62728", "out": "#ff7f0e", "excluded": "#9467bd"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _depth_order(rec):
    loc = rec.get_channel_locations()
    order = np.lexsort((loc[:, 0], loc[:, 1]))
    return order, loc[order, 1]


def raw_vs_preprocessed(path: Path, sid: str, raw_uV, pre, labels: dict, i_mid: int, to_time, seconds: float = 0.2) -> str:
    """Two heat maps of the same `seconds` of data, channels by depth: as recorded vs preprocessed.

    `raw_uV` and `pre` are sample-aligned (the same window); `to_time` maps sample index -> session seconds."""
    plt = _plt()
    fs = raw_uV.get_sampling_frequency()
    i0 = int(np.clip(i_mid, 0, max(0, raw_uV.get_num_samples() - int(seconds * fs))))
    i1 = min(raw_uV.get_num_samples(), i0 + int(seconds * fs))
    t = float(to_time(i0))
    o_raw, d_raw = _depth_order(raw_uV)
    raw = raw_uV.get_traces(start_frame=i0, end_frame=i1)[:, o_raw]
    raw = raw - np.median(raw, axis=0, keepdims=True)
    o_pre, d_pre = _depth_order(pre)
    clean = pre.get_traces(start_frame=i0, end_frame=i1)[:, o_pre]
    lim = 5 * float(np.median(np.median(np.abs(clean), axis=0) / 0.6745) + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True, constrained_layout=True)
    ext_ms = (i1 - i0) / fs * 1000
    for ax, data, depth, title in ((axes[0], raw, d_raw, "As recorded (median per channel removed)"),
                                   (axes[1], clean, d_pre, "Preprocessed (input of spike sorting)")):
        im = ax.imshow(data.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=(0, ext_ms, depth.min(), depth.max() if depth.max() > depth.min() else depth.min() + 1),
                       interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"time from {t:.2f} s (ms)")
    axes[0].set_ylabel("depth along the probe (um)")
    ch_ids = np.asarray(raw_uV.channel_ids)[o_raw]
    for c, y in zip(ch_ids, d_raw):
        lab = labels.get(str(c), "good")
        if lab != "good":
            axes[0].plot([-0.012 * ext_ms], [y], marker="s", ms=3, color=LABEL_COLOURS.get(lab, "k"), clip_on=False)
    fig.colorbar(im, ax=axes, label="uV", shrink=0.8)
    fig.suptitle(f"{sid}: raw vs preprocessed traces", fontsize=12)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return (f"{seconds * 1000:.0f} ms of every channel, sorted by depth, before and after preprocessing (same colour scale, "
            "±5 times the typical noise). Raw data show slow offsets and stripes shared by all channels (common noise); after "
            "filtering and re-referencing only spikes should remain, as short dark (negative) blips spanning a few neighbouring "
            "channels. Squares left of the raw panel mark channels removed as bad (blue dead, red noisy, orange outside the brain, "
            "purple excluded by hand).")


def bad_channels(path: Path, sid: str, filtered, labels: dict, noise_uV: dict) -> str:
    plt = _plt()
    o, depth = _depth_order(filtered)
    ids = [str(c) for c in np.asarray(filtered.channel_ids)[o]]
    noise = np.array([noise_uV.get(c, np.nan) for c in ids])
    labs = [labels.get(c, "good") for c in ids]
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    for lab in LABEL_COLOURS:
        m = np.array([x == lab for x in labs])
        if m.any():
            ax.scatter(noise[m], depth[m], s=10 if lab == "good" else 22, color=LABEL_COLOURS[lab],
                       label=f"{lab} ({int(m.sum())})", zorder=3 if lab != "good" else 2)
    ax.set_xlabel("noise level after filtering (uV, median absolute deviation)")
    ax.set_ylabel("depth along the probe (um)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{sid}: bad channels ({sum(lab != 'good' for lab in labs)} of {len(labs)})", fontsize=11)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return ("Noise of every channel along the probe, coloured by what bad-channel detection decided (coherence with "
            "neighbouring channels and high-frequency power, as in the IBL and AIND pipelines). Dead channels are flat or "
            "unlike their neighbours, noisy channels have excess high-frequency power, and channels at the top of the probe "
            "that lost coherence are outside the brain. These are removed before sorting. A healthy recording has a few "
            "percent bad channels; many more suggests a damaged probe or a reference problem.")


def drift_raster(path: Path, sid: str, rec, peaks, peak_locations, motion, applied: bool, to_time, max_points: int = 150_000) -> str:
    """`motion` in `rec`'s own clock (for the correction); `to_time` maps sample index -> session seconds (x axis)."""
    plt = _plt()
    from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

    rng = np.random.default_rng(0)
    idx = np.arange(len(peaks))
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, max_points, replace=False))
    p, loc = peaks[idx], peak_locations[idx]
    times = to_time(p["sample_index"])
    amp = np.abs(p["amplitude"])
    vmax = np.percentile(amp, 95) if len(amp) else 1
    panels = [("as recorded", loc["y"])]
    if motion is not None:
        corrected = correct_motion_on_peaks(p, loc, motion, rec)
        panels.append(("after motion correction" + ("" if applied else " (estimated; not applied to the saved data)"), corrected["y"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.5), sharey=True, squeeze=False, constrained_layout=True)
    order = np.argsort(amp)
    for ax, (title, y) in zip(axes[0], panels):
        ax.scatter(times[order], y[order], c=amp[order], s=1, cmap="Greys", vmin=0, vmax=vmax, alpha=0.6, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s, session clock)")
    axes[0][0].set_ylabel("estimated spike depth (um)")
    fig.suptitle(f"{sid}: drift raster ({len(peaks):,} detected spikes, {len(idx):,} shown)", fontsize=12)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return ("Every detected spike as a dot at its time and estimated depth on the probe (darker = larger). Neurons show up "
            "as horizontal bands. If the bands bend or shift over time the brain moved relative to the probe (drift); the right "
            "panel shows the same spikes after removing the estimated motion, where the bands should be straight. Sudden steps "
            "affecting all depths are probe or animal movements; slow sagging is tissue relaxation.")


def motion_traces(path: Path, sid: str, motion) -> tuple[str, dict]:
    plt = _plt()
    disp = np.asarray(motion.displacement[0])          # (time bins, spatial bins)
    tb = np.asarray(motion.temporal_bins_s[0])
    sb = np.asarray(motion.spatial_bins_um)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, gridspec_kw={"width_ratios": [3, 1]})
    cmap = plt.get_cmap("viridis")
    for j in range(disp.shape[1]):
        axes[0].plot(tb, disp[:, j], color=cmap(j / max(1, disp.shape[1] - 1)), lw=1,
                     label=f"{sb[j]:.0f} um" if disp.shape[1] <= 12 else None)
    axes[0].set_xlabel("time (s, session clock)")
    axes[0].set_ylabel("estimated displacement (um)")
    axes[0].axhline(0, color="k", lw=0.5)
    if disp.shape[1] <= 12:
        axes[0].legend(fontsize=7, title="depth", title_fontsize=7, ncol=2)
    ptp = disp.max(axis=0) - disp.min(axis=0)
    axes[1].barh(sb, ptp, height=(sb[1] - sb[0]) * 0.8 if len(sb) > 1 else 20, color="#555")
    axes[1].set_xlabel("drift range (um, max - min)")
    axes[1].set_ylabel("depth (um)")
    fig.suptitle(f"{sid}: estimated motion (DREDge)", fontsize=12)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    stats = {"max_abs_displacement_um": float(np.abs(disp).max()) if disp.size else 0.0,
             "median_drift_range_um": float(np.median(ptp)) if ptp.size else 0.0,
             "max_drift_range_um": float(ptp.max()) if ptp.size else 0.0,
             "n_spatial_bins": int(disp.shape[1]), "temporal_bin_s": float(tb[1] - tb[0]) if len(tb) > 1 else None}
    caption = (f"How far the tissue moved along the probe over time, estimated with DREDge, one line per depth window "
               f"(left), and the total range per depth (right). Here the largest range is {stats['max_drift_range_um']:.0f} um. "
               "Under about 10 um the sorter hardly notices; tens of micrometres (several contact spacings) is why drift "
               "correction matters, and a range much larger than that, or sudden jumps, can split one neuron into several units.")
    return caption, stats

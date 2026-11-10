# ephys-prep

This app is built on **[SpikeInterface](https://github.com/SpikeInterface/spikeinterface)**, developed by
Alessio Buccino, Samuel Garcia and the SpikeInterface team (Centre de Recherche en Neuroscience de Lyon / CNRS,
Allen Institute for Neural Dynamics, CatalystNeuro, University of Edinburgh and others;
[Buccino et al. 2020, eLife](https://doi.org/10.7554/eLife.61834)), and follows the preprocessing of the
**[AIND ephys pipeline](https://github.com/AllenNeuralDynamics/aind-ephys-pipeline)** by Alessio Buccino, Arjun
Sridhar, David Feng, Karel Svoboda and Joshua Siegle at the Allen Institute for Neural Dynamics
([Buccino et al. 2026, eLife](https://doi.org/10.7554/eLife.110170)), which in turn follows the
**[International Brain Laboratory's spike sorting pipeline](https://doi.org/10.6084/m9.figshare.19705522)**
(phase shift, bad-channel detection, destriping). Drift is estimated with **[DREDge](https://github.com/evarol/dredge)**,
developed by Charlie Windolf, Erdem Varol, Liam Paninski and colleagues at Columbia University
([Windolf et al. 2025, Nature Methods](https://doi.org/10.1038/s41592-025-02614-5)), as implemented in
SpikeInterface ([Garcia et al. 2024, eNeuro](https://doi.org/10.1523/ENEURO.0229-23.2023)). Recordings are read
with [Neo](https://github.com/NeuralEnsemble/python-neo), [probeinterface](https://github.com/SpikeInterface/probeinterface)
and [PyNWB](https://github.com/NeurodataWithoutBorders/pynwb). **Pennsieve packages these methods as a platform app; it
did not develop them.** The demo data are the International Brain Laboratory's Brain Wide Map recordings on DANDI
([DANDI:000409](https://doi.org/10.48324/dandi.000409/0.260309.1324)) and Laquitaine et al.'s simulated ground-truth
recording ([DANDI:001250](https://doi.org/10.48324/dandi.001250/0.261108.1908)).

**What it does.** Before spikes can be sorted, a raw extracellular recording has to be cleaned: slow signals
(LFP) and offsets removed, broken channels thrown out, noise that all channels share subtracted, and the brain's
movement relative to the probe (drift) measured. `ephys-prep` does this the way the Allen Institute and the IBL do
it, and writes a recording that any spike sorter can read, with QC figures that show what changed. It is the
first step of the **Neuropixels Spike Sorting** workflow:

`ephys-prep → spike-sort → ephys-curate → neuro-report`

It works on any lab's recordings (SpikeGLX, Open Ephys, Intan, NWB from DANDI, raw binary) and any probe, not
only Neuropixels.

## Method (per session = one probe of one recording)

1. **Read** lazily with SpikeInterface (nothing is loaded into memory): SpikeGLX `*.ap.bin` + `.meta` (or IBL's
   compressed `*.ap.cbin`), Open Ephys (binary or legacy), Intan `*.rhd`/`*.rhs`, NWB `ElectricalSeries` in
   `acquisition/` (spike band, ≥ 10 kHz; LFP series are skipped), SpikeInterface folders, or a raw binary file
   with a JSON sidecar. Each probe of a recording is its own session. Probe geometry comes from the file
   (SpikeGLX IMRO table, Open Ephys settings, NWB `rel_x`/`rel_y`), else from a probeinterface JSON next to the
   recording, else a single 20 µm column is assumed (with a warning). Brain areas (NWB electrodes `location`) are
   kept as the channel property `brain_area`.
2. **Clock.** Times are seconds on the session clock: the NWB file's own `timestamps` when it has them (IBL raw
   NWB: the probe synchronised to the NIDQ clock), else the start time plus samples / rate. Explicit timestamps
   are checked and classified:
   - **pauses** (one positive gap with a normal rate on both sides, e.g. a paused acquisition) are kept: no data
     are dropped, the timestamps carry the gap (the sorter sees the samples back to back);
   - **glitches** (a few samples going backwards, repeating or jumping, within 1 s, after which the timestamps are
     back on their line) are repaired by interpolating between the good neighbours;
   - **rate faults** (a stretch whose rate over 1 s blocks is more than 0.1 % off the median, such as the last
     168 s of IBL's raw file for this session, where a sync fault squeezes 168 s into 84 s) and unrepairable
     glitches are dropped by default (`BAD_TIMESTAMPS=drop`: the longest good stretch is kept; `keep` uses them).
   All of it is recorded in `session.json` (`timestamp_problems`) and `summary.json`.
   **Sampling rate**: with explicit timestamps the recording's `sampling_frequency` is their effective rate over
   the session (samples / time, pauses excluded): 30,000.48 Hz for the IBL probe, where SpikeInterface's own
   estimate from the first 1,000 intervals is 29,997.0 Hz (70 ms off after 10 min). The times file
   (`times_cached_seg0.raw`) is the authority; use `recording.get_times()` / `sample_index_to_time`, not
   samples / rate, to get seconds. The difference from the nominal rate (30 kHz for Neuropixels) is recorded and
   reported when above 0.1 %.
3. **Window** (`T_START`, `DURATION`) on the session clock; never re-zeroed.
4. **Phase shift** (Neuropixels): the probe's ADCs sample channels in turn, up to 0.85 samples apart (NP 1.0);
   this delay is undone (IBL `fshift`, SpikeInterface `phase_shift`, 100 ms margin) so a spike reaches neighbouring
   channels at the same time. The ADC timing comes from SpikeGLX / Open Ephys or, for NWB, from the probe part
   number in the device (probeinterface's ProbeTable copy).
5. **High-pass 300 Hz** (or band-pass 300-6000 Hz), in µV, float32.
6. **Bad channels** (AIND parameters): SpikeInterface `detect_bad_channels(method="coherence+psd")`, IBL's
   method: dead channels (low coherence with neighbours), noisy channels (high-frequency power), and channels
   outside the brain at the top of the probe; removed. More than `MAX_BAD_CHANNEL_FRACTION` bad fails the session.
   Probes with fewer than 32 channels use a noise-level (MAD) rule instead.
7. **Re-reference**: common median reference over all channels (AIND default), or IBL destriping
   (`highpass_spatial_filter` with automatic gain control: removes stripes shared by nearby channels).
8. **Drift (motion)** is always estimated: DREDge on spikes (`MOTION_SOURCE=ap`, `dredge_fast`: peaks detected at
   8 × noise, localised by grid convolution, non-rigid windows of 10 % of the probe length as in AIND), or on the
   LFP (`MOTION_SOURCE=lfp`: the probe's LF stream, band-passed 0.5-250 Hz, resampled to 250 Hz, second spatial
   derivative, rigid; better when units are sparse or the motion is fast, as in human intraoperative and NHP
   recordings; Neuropixels 1.0 AP data have no LFP, so it falls back to spikes there). The estimate is saved as a
   SpikeInterface `Motion` on the session clock, with its largest drift range (µm) and whether that exceeds
   `MOTION_THRESHOLD_UM` (10 µm).
   **Whether to interpolate** follows Garcia et al. 2026 and AIND: **Kilosort4 keeps its own drift correction**
   (spike-sort's gpu variant, `nblocks=5`), so the data are not changed here (default `MOTION_CORRECTION=estimate`);
   the CPU sorters (Lupin, SpykingCircus2) **interpolate the recording with this estimate in spike-sort, and only
   when the drift exceeds the threshold** (below ~10 µm interpolation costs more accuracy than it recovers).
   Correcting twice is never done. `session.json` records the estimate, the threshold and the decision
   (`motion_above_threshold`, `motion_decision`). `MOTION_CORRECTION=apply` interpolates here instead (kriging)
   when above the threshold, for sorters outside this workflow; `session.json` then says `motion_corrected: true`
   and spike-sort corrects nothing (Kilosort4 `nblocks=0`).
9. **Write** int16 (the source's own µV per bit; for float sources 1/20 of the median noise), clipped, never
   wrapped, as a SpikeInterface `binary_folder` with the probe, `brain_area` and the session-clock timestamps.

## Ports

| Port | Kind | Files | Notes |
|---|---|---|---|
| `recording` | input | raw recordings (see Method 1), optional `sessions.csv`, optional probeinterface JSON | Found by name and content anywhere below, following symlinks. |
| `recording` | output | `OUTPUT_DIR/recording/ephys-prep/...` (below) | The data and the small files. |
| `qc` | output | `OUTPUT_DIR/qc/ephys-prep/`: `sessions.csv`, `summary.json`, `provenance.json`, `<session_id>/session.json` and the four QC PNGs | Only the small files, at the same relative paths, so the report (a Lambda step) can be linked without the data. |

## Parameters

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `T_START` | number | (empty) | Window start in seconds on the session clock; empty = recording start. |
| `DURATION` | number | `0` | Window length in seconds; `0` = to the end. |
| `STREAM` | string | (empty) | Comma-separated parts of stream names to keep (`imec0`, `ProbeA`, `Probe01`); empty = every probe. |
| `FILTER` | string | `highpass` | `highpass` (AIND, IBL) or `bandpass`. |
| `FREQ_MIN` / `FREQ_MAX` | number | `300` / `6000` | Filter corners (Hz); `FREQ_MAX` only for `bandpass`. |
| `PHASE_SHIFT` | string | `auto` | `auto` (when the ADC timing is known), `true`, `false`. |
| `BAD_CHANNELS` | string | `auto` | `auto` (dead, noisy, outside the brain), `dead-noisy` (keep "outside" channels: simulations, probes fully in tissue), `none`. |
| `EXCLUDE_CHANNELS` | string | (empty) | Extra channel ids to remove. |
| `MAX_BAD_CHANNEL_FRACTION` | number | `0.5` | A session with more bad channels fails. |
| `REFERENCE` | string | `cmr` | `cmr` (global median), `destripe` (IBL), `none`. |
| `MOTION_CORRECTION` | string | `estimate` | `estimate` (default: estimate only; spike-sort decides), `apply` (interpolate here when above the threshold; spike-sort then corrects nothing), `none`. |
| `MOTION_THRESHOLD_UM` | number | `10` | Drift (largest displacement range, µm) above which interpolation is worth it; `0` = always. |
| `MOTION_SOURCE` | string | `ap` | `ap` (DREDge on spikes) or `lfp` (DREDge on the LFP; human / NHP, sparse units). |
| `MOTION_PRESET` | string | `dredge_fast` | Spike-based preset: `dredge_fast` or `dredge`. |
| `BAD_TIMESTAMPS` | string | `drop` | Broken explicit timestamps: `drop` (keep the longest good stretch) or `keep`. |
| `THREADS` | integer | `0` | Worker processes; `0` = the container's CPU quota. |
| `SEED` | integer | `42` | Seed of the random chunks for noise and bad-channel estimates. |

Bad parameters exit 2; no recording found, or every session failed, exit 1; a failed session is logged and skipped.

## Outputs (`OUTPUT_DIR/recording/ephys-prep/`; the small files also in `OUTPUT_DIR/qc/ephys-prep/`)

| File | Contents |
|---|---|
| `<session_id>/recording/` | SpikeInterface `binary_folder`: `traces_cached_seg0.raw` (int16, samples × channels), `times_cached_seg0.raw` (float64 session-clock time of every sample, when the source had timestamps; else `t_start` in `binary.json`), `probegroup.json`, `properties/` (`brain_area` when known, gains). Open with `spikeinterface.load(folder)`. |
| `<session_id>/motion/` | SpikeInterface `Motion` (`Motion.load`): displacement (µm) per time bin (1 s, session clock) and depth window. |
| `<session_id>/session.json` | `session_id, source, source_format, stream, probe {model, serial, part_number, geometry_source}, sampling_frequency, t_start, t_stop, duration_s, n_channels, n_channels_source, bad_channels [{id, reason}], motion_corrected, motion_estimated, motion_method, motion_source, motion_threshold_um, motion_above_threshold, motion_decision, motion {max_abs_displacement_um, median_drift_range_um, max_drift_range_um, estimate_params}, time_source, effective_sampling_frequency / nominal_sampling_frequency / clock_difference_percent` (explicit timestamps), `timestamp_problems` (when broken), `gain_to_uV, preprocessing` (every step with its parameters), the source NWB's `session_start_time, identifier, session_description, timestamps_reference_time, institution, lab, experimenter, experiment_description, keywords, protocol, nwb_session_id` (the source's own `session_id`) and the full `subject` (every field of `general/subject`: subject_id, sex, species, strain, genotype, date_of_birth / age, weight, description), and `sheet` (the session's `sessions.csv` columns). |
| `<session_id>/<session_id>_raw_traces.png` | 0.2 s of every channel by depth, raw vs preprocessed. |
| `<session_id>/<session_id>_bad_channels.png` | Noise per channel along the probe, coloured by bad-channel label. |
| `<session_id>/<session_id>_drift_raster.png` | Detected spikes by time and depth, before and after the motion estimate. |
| `<session_id>/<session_id>_motion.png` | Estimated displacement over time per depth window, and the drift range per depth. |
| `sessions.csv` | `session_id, path, subject_id, condition, group, status, source_format, stream, probe_model, t_start, t_stop, duration_s, n_channels_source, n_channels, n_bad_channels, motion_corrected, max_drift_range_um, n_peaks, seconds, error` + extra sheet columns. |
| `summary.json` | Per session: duration, channels, bad channels, drift, peaks per second, time source and clock problems; the figures with captions. |
| `provenance.json` | App and tool versions, parameters, inputs, timings, peak memory, citations (DOIs). |

**Session ids**: from `sessions.csv` (`session_id`, `path` matched against file and folder names), else
`sub-<a>_ses-<b>` from the file or a parent folder (a UUID session cut to 8 characters:
`sub-NR-0031_ses-caa5dddc`), else the file stem (folder name for SpikeGLX / Open Ephys). When a recording has
more than one probe, `_probe00`, `_probe01`, ... is appended (from the stream name: `imec1` → `_probe01`,
`ProbeB` → `_probe01`), also when `STREAM` keeps only one of them.

## Results on the demo data

**IBL, `sub-NR-0031_ses-caa5dddc`, Probe01** (NP 1.0, 384 channels, 10 min, 899.67-1499.67 s on the session
clock; DANDI:000409). Phase shift applied (NP 1.0 part
number `PRB_1_4_0480_1_C` from the NWB device), 1 bad channel (`AP191`, dead), 383 channels written (13 GB int16).
Timestamps: effective rate 30,000.48 Hz vs 30,000 Hz nominal (+0.0016 %, below the 0.1 % reporting threshold; no
broken stretches in this cut); the written recording's `sampling_frequency` is that effective rate. DREDge found
269,272 spikes (449 per s) and very little drift: median range 2.4 µm, largest 8.4 µm, below the 10 µm threshold, so the CPU sorters will not interpolate and Kilosort4 corrects on its
own. Brain areas from IBL's histology (17, from V1 through hippocampus to thalamus and midbrain) are kept on the
channels. Output: the workflow repo's `data/work/ibl/ephys-prep/`.

**Ground truth** (DANDI:001250 `sub-demo-npx-bio-evoked`: simulated NP 1.0 shank, 120 channels, 20 kHz,
float32 µV, 60 s; `BAD_CHANNELS=dead-noisy`): the simulation's float32 µV were written as int16 at 0.17 µV/bit (1/20 of
the median noise); no bad channels; drift 0.2 µm (none was simulated). With `BAD_CHANNELS=auto` the "outside the
brain" rule flags 38 quiet channels at the top of this simulated shank (see Limitations). Output:
`data/work/ground-truth/ephys-prep/`.

**Synthetic fixture** (`tests/make_fixture.py`, 64 channels, 12 planted units, a dead and a noisy channel, 15 µm
of drift, a 60 Hz + 3 Hz common signal): both planted bad channels are found, the common signal is removed
(noise 9 µV), DREDge recovers the drift (12.9 µm median range for 15 µm planted, over 20 s), timestamps that run
0.2 % fast are kept as they are (and reported), and a broken stretch of timestamps is dropped.

**Vendor formats** (opt-in test `EPHYS_FORMAT_DATA`, files from NEO's `ephy_testing_data` on GIN, not in the
repository): SpikeGLX (a real NP 1.0 `.ap.meta`, `PRB_1_4_0480_1`, with 12 s of synthetic traces: 384 channels,
phase shift applied), Open Ephys binary (a real v0.6 Neuropix-PXI `settings.xml` / `structure.oebin`, 10 s of
synthetic traces: the AP stream is read, LFP and NI-DAQ streams skipped, the sample-number clock kept at
727.9 s) and Intan (a real `.rhd` file, 192 amplifier channels, 1 s; no geometry, so one column is assumed and
flagged) all process to `status ok`. Raw NWB was tested on the IBL file and the fixtures.

## Runtime

Measured on an M-series Mac. **Native arm64** (6 worker processes): the IBL 10 min probe took **9-18 min**
depending on the load from other jobs on the machine (DREDge about two thirds, writing most of the rest); peak memory 4.6 GB in the main process plus ~0.5-1.2 GB per
worker; 120 s took 2 min. The ground-truth file (120 channels, 60 s) took 1.6 min. **Docker, linux/amd64 under
emulation**, Lambda-style (read-only, uid 65534, default 64 MB `/dev/shm`, 2 CPUs, 4 GB): 30 s of the IBL probe
completed (main process 1.6 GB; cgroup peak 3.6 GB including page cache), and also with no `/dev/shm` at all
(`--ipc none`, as on Lambda: SpikeInterface then runs on threads instead of worker processes, which need POSIX
semaphores; 3 min, 3.1 GB peak). SpikeInterface's shared memory is file-backed in `$TMPDIR/shm`
(`processor/fileshm.py`), so the 64 MB `/dev/shm` of containers is never a limit. Under QEMU (5-10 × slower than native): the fixture (64 channels, 10 s) in the Lambda-style read-only
container (2 CPUs, 4 GB) took 5-10 min with 1.9 GB peak; 60 s of the IBL probe (2 CPUs, 4 GB, 2 workers) took 12 min (main process 1.5 GB; the cgroup peak reached its 4 GB limit with page cache of the 6.6 GB NWB file).

Memory scales with the number of workers: each spawned worker holds the stack (~350 MB) and a few 1 s chunks of
384 channels; peak detection and localisation reach ~1.2 GB per worker. The app caps the workers at
(container memory − 1.5 GB) / 1.2 GB. Chosen: **8 vCPU, 16 GB** (8 workers fit), timeout 6 h: about 20-30 min per
10 min of a Neuropixels probe on x86 (est.), i.e. ~2-3 h per probe-hour. Disk: the output is the size of the int16
recording (~83 GB per probe-hour). This is too big for a Lambda step (> 10 GB, > 15 min) except for short windows
of small probes.

## Limitations

- Bad-channel detection's "outside the brain" rule assumes the top of the probe is the end that leaves the brain
  (Neuropixels, inserted tip first). On a simulation or a probe fully in tissue it can flag quiet channels at the
  top (it flagged 38 of 120 on the DANDI:001250 simulation): use `BAD_CHANNELS=dead-noisy` there.
- A probe with fewer than 16 channels or under 150 µm long gets no drift estimate (tetrodes, sparse arrays). DREDge
  assumes motion along the probe (y) only; the LFP estimate is rigid (one displacement for the whole probe).
- The 10 µm threshold is a rule of thumb from simulations (Garcia et al. 2026), not a law: with fast, non-rigid
  motion consider `MOTION_THRESHOLD_UM=0`.
- Multi-shank probes are processed as one probe (one median reference over all shanks, one motion estimate). Split
  shanks with `STREAM` / a probe JSON if they move independently.
- Intan "traditional" files that split one session into several `.rhd` files are read as separate sessions.
- A recording without probe geometry (Intan or raw binary without a probeinterface JSON) is processed as one 20 µm
  column, with a prominent warning in the log and `summary.json`, and no drift estimate: add a `<name>.probe.json`.
- The phase shift needs the probe's ADC layout: known for Neuropixels from SpikeGLX/Open Ephys metadata or an NWB
  device naming the part number; other NWB Neuropixels files are not shifted (logged).
- Nothing here decides what is a neuron: that is `spike-sort` and `ephys-curate`.

## Licences

Apache-2.0 (this repository). The image bundles SpikeInterface, probeinterface (MIT), Neo, PyNWB, HDMF (BSD-3),
PyTorch CPU (BSD-3), mtscomp (BSD-3) and others: see [`THIRD_PARTY.md`](THIRD_PARTY.md). No GPL code.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author lists in
[`CITATION.cff`](CITATION.cff).

- **SpikeInterface**: Buccino AP, Hurwitz CL, Garcia S, Magland J, Siegle JH, Hurwitz R, Hennig MH. SpikeInterface,
  a unified framework for spike sorting. *eLife* 2020;9:e61834. doi:10.7554/eLife.61834. PMID 33170122.
- **AIND pipeline**: Buccino AP, Sridhar A, Feng D, Svoboda K, Siegle JH. Efficient and reproducible pipelines for
  spike sorting large-scale electrophysiology data. *eLife* 2026;15:RP110170. doi:10.7554/eLife.110170. PMID 42554034.
- **IBL preprocessing** (phase shift, bad channels, destriping): International Brain Laboratory. Spike sorting
  pipeline for the International Brain Laboratory. figshare 2024. doi:10.6084/m9.figshare.19705522.
- **Common median reference**: Rolston JD, Gross RE, Potter SM. Common median referencing for improved action
  potential detection with multielectrode arrays. 2009 Annual International Conference of the IEEE Engineering in
  Medicine and Biology Society, 1604-1607. doi:10.1109/IEMBS.2009.5333230. PMID 19964004.
- **DREDge**: Windolf C, Yu H, Paulk AC, et al. DREDge: robust motion correction for high-density extracellular
  recordings across species. *Nature Methods* 2025;22:788-800. doi:10.1038/s41592-025-02614-5. PMID 40050699.
- **Motion correction in SpikeInterface**: Garcia S, Windolf C, Boussard J, Dichter B, Buccino AP, Yger P. A modular
  implementation to handle and benchmark drift correction for high-density extracellular recordings. *eNeuro*
  2024;11:ENEURO.0229-23.2023. doi:10.1523/ENEURO.0229-23.2023. PMID 38238082.
- **Neo**: Garcia S, Guarino D, Jaillet F, et al. Neo: an object model for handling electrophysiology data in
  multiple formats. *Front Neuroinform* 2014;8:10. doi:10.3389/fninf.2014.00010. PMID 24600386.
- **NWB**: Rübel O, Tritt A, Ly R, et al. The Neurodata Without Borders ecosystem for neurophysiological data
  science. *eLife* 2022;11:e78362. doi:10.7554/eLife.78362. PMID 36193886.
- **Neuropixels**: Jun JJ, Steinmetz NA, Siegle JH, et al. Fully integrated silicon probes for high-density
  recording of neural activity. *Nature* 2017;551:232-236. doi:10.1038/nature24636. PMID 29120427. Steinmetz NA,
  Aydin C, Lebedeva A, et al. Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain
  recordings. *Science* 2021;372:eabf4588. doi:10.1126/science.abf4588. PMID 33859006.
- **Demo data**: International Brain Laboratory. IBL - Brain Wide Map. DANDI Archive 2026.
  doi:10.48324/dandi.000409/0.260309.1324; International Brain Laboratory, et al. A brain-wide map of neural activity
  during complex behaviour. *Nature* 2025;645:177-191. doi:10.1038/s41586-025-09235-0. PMID 40903598.
  Laquitaine S, et al. Spike sorting biases and information loss in a detailed cortical model. DANDI Archive 2026.
  doi:10.48324/dandi.001250/0.261108.1908; bioRxiv 2024. doi:10.1101/2024.12.04.626805.
- **Software**: [probeinterface](https://github.com/SpikeInterface/probeinterface),
  [PyNWB](https://github.com/NeurodataWithoutBorders/pynwb), [mtscomp](https://github.com/int-brain-lab/mtscomp),
  [PyTorch](https://github.com/pytorch/pytorch).

## Acknowledgements

The SpikeInterface team and contributors; the Allen Institute for Neural Dynamics for the open AIND ephys
pipeline and its documented defaults; the International Brain Laboratory for its spike sorting white paper,
open data and tools (mtscomp, destriping); the DREDge authors (Paninski and Varol labs); the Neurodata Without
Borders project and the DANDI Archive (BRAIN Initiative) for the data standard and hosting the demo data; Steeve
Laquitaine and colleagues for the ground-truth simulation.

# sample-input

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/ -> data/output/
```

`tests/make_fixture.py` writes (numpy only; `--nwb` also writes NWB files with pynwb):

- `input/sub-fixture_ses-01/sub-fixture_ses-01_ecephys.bin` + `.json` sidecar + `.probe.json` (probeinterface):
  64 channels (2 columns × 32 rows, 20 µm pitch), 30 kHz, int16 at 2.34 µV/bit, starting at 100 s on the session
  clock. 12 planted units (80-220 µV), 15 µm of slow drift, a 3 Hz + 60 Hz common signal, channel 5 dead and
  channel 40 noisy. Ground truth: `ground_truth/sub-fixture_ses-01_gt.npz`.
- `nwb/sub-fixnwb_ses-01_ecephys.nwb`: the same as an NWB `ElectricalSeries` with explicit `timestamps` from
  1000 s on a clock running 0.2 % fast, electrodes with `rel_x`/`rel_y` and `location`.
- `nwb2/twoprobes.nwb`: two AP `ElectricalSeries` (Probe00, Probe01) and an LFP one → sessions `twoprobes_probe00`,
  `twoprobes_probe01`.
- `nwb3/sub-badts_ses-01_ecephys.nwb`: timestamps that jump to an apparent 60 kHz for the last 5 s (like IBL's raw
  file after its sync broke).

The binary sidecar format (any `<name>.bin|.dat|.raw` with `<name>.json`):

```json
{"sampling_frequency": 30000, "num_channels": 64, "dtype": "int16", "gain_to_uV": 2.34375,
 "offset_to_uV": 0, "t_start": 100.0, "time_axis": 0, "file_offset": 0}
```

## Real data

Any SpikeGLX, Open Ephys, Intan or NWB recording. The demo data of the workflow:

- **IBL Brain Wide Map, DANDI:000409** (CC-BY-4.0, doi:10.48324/dandi.000409/0.260309.1324): session
  `sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614`, raw file
  `sub-NR-0031/sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614_desc-raw_ecephys.nwb` (52.5 GB, two NP 1.0
  probes, AP and LF series with timestamps). The workflow's demo uses a 10 min cut of Probe01 (899.67-1499.67 s),
  `sub-NR-0031_ses-caa5dddc_desc-raw-10min_ecephys.nwb` (6.6 GB), made with the workflow repo's
  `scripts/stage-demo-data.py ibl`. On the full file, `STREAM=Probe01 T_START=899.67 DURATION=600` gives the same
  window; note that its timestamps after 2001.74 s are broken (dropped by default).
- **Ground truth, DANDI:001250** (CC-BY-4.0, doi:10.48324/dandi.001250/0.261108.1908):
  `sub-demo-npx-bio-evoked/sub-demo-npx-bio-evoked_ecephys.nwb` (0.6 GB, 120 channels, 20 kHz, 60 s, simulated);
  use `BAD_CHANNELS=dead-noisy`.
- **Kilosort4 tutorial data** (IBL, CC-BY): `ZFM-02370_mini.imec0.ap.short.bin` + `.meta`, 90 s of NP 1.0
  (https://osf.io/download/67effd64f74150d8738b7f34/), SpikeGLX format.

# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image contains the following third-party components, installed from
PyPI at the pinned versions in `requirements.txt` / `requirements-torch.txt`; their licences apply to the image.
**Nothing in the image is GPL.** A few components are MPL-2.0 (file-level copyleft; shipped unmodified).
Licences were read from each installed package's metadata in the built image (`importlib.metadata`), and from the
project's repository where the metadata had none (quantities).

| Component | What it does here | Authors / copyright holder | Licence | Source |
|---|---|---|---|---|
| SpikeInterface 0.105.4 | Reading, preprocessing (phase shift, filters, bad channels, references, destriping), DREDge motion estimation, binary folders | Alessio Buccino, Samuel Garcia and the SpikeInterface contributors | MIT | https://github.com/SpikeInterface/spikeinterface |
| probeinterface 0.4.2 | Probe geometry, Neuropixels ProbeTable (ADC timing) | Samuel Garcia, Alessio Buccino, Chris Halcrow and contributors | MIT | https://github.com/SpikeInterface/probeinterface |
| Neo 0.14.6 | SpikeGLX, Open Ephys, Intan readers | Neo authors and contributors (NeuralEnsemble) | BSD-3-Clause | https://github.com/NeuralEnsemble/python-neo |
| PyNWB 4.2.0, HDMF 6.2.0 | NWB reading (via SpikeInterface) | Andrew Tritt, Ryan Ly, Oliver Ruebel, Ben Dichter, Matthew Avaylon, Stephanie Prince and contributors (LBNL / NWB) | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb, https://github.com/hdmf-dev/hdmf |
| h5py 3.16.0 | HDF5 access (bundles the HDF5 library, BSD-style) | Andrew Collette and contributors | BSD-3-Clause | https://www.h5py.org |
| PyTorch 2.14.1 (CPU build) | DREDge motion estimation | PyTorch Team (Meta and contributors) | BSD-3-Clause (with bundled Apache-2.0, BSD-2-Clause, BSL-1.0, MIT components) | https://github.com/pytorch/pytorch |
| mtscomp 1.0.2 | Reading IBL compressed `.cbin` | Cyrille Rossant (International Brain Laboratory) | BSD-3-Clause | https://github.com/int-brain-lab/mtscomp |
| NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1 | Arrays, filters, statistics | NumPy, SciPy, pandas, scikit-learn developers | BSD-3-Clause | numpy.org, scipy.org, pandas.pydata.org, scikit-learn.org |
| numba 0.68.1, llvmlite 0.50.0 | JIT kernels of SpikeInterface | Anaconda, Inc. and contributors | BSD-2-Clause; llvmlite BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | https://numba.pydata.org |
| Matplotlib 3.11.2 | QC figures | Matplotlib Development Team | Matplotlib licence (PSF-based, BSD-compatible) | https://matplotlib.org |
| zarr 2.18.7, numcodecs 0.15.2 | Dependencies of SpikeInterface | zarr developers | MIT | https://github.com/zarr-developers |
| quantities 0.16.4 | Dependency of Neo | python-quantities developers | BSD-style ("Quantities only uses BSD compatible code", doc/user/license.rst) | https://github.com/python-quantities/python-quantities |
| certifi, tqdm | Dependencies | — | MPL-2.0 (tqdm: MPL-2.0 AND MIT) | PyPI |
| Other transitive dependencies (attrs, cloudpickle, contourpy, cycler, Deprecated, fasteners, filelock, fonttools, fsspec, idna, Jinja2, joblib, jsonschema, kiwisolver, MarkupSafe, mpmath, narwhals, networkx, packaging, pillow, platformdirs, pydantic, pyparsing, python-dateutil, referencing, requests, rpds-py, ruamel.yaml, sympy, threadpoolctl, typing_extensions, tzdata, urllib3, wrapt, ...) | Versions in `requirements.txt` | their authors | MIT / BSD / Apache-2.0 / PSF-2.0 / MIT-CMU (Pillow) | PyPI |
| python:3.12.16-slim-bookworm | Base image (Debian 12 + CPython 3.12) | Python Software Foundation; Debian | PSF (Python); Debian packages under their own free licences | https://hub.docker.com/_/python |

## Methods used, not bundled as separate code

- The **AIND ephys pipeline** (Allen Institute for Neural Dynamics, MIT,
  https://github.com/AllenNeuralDynamics/aind-ephys-pipeline): its `default_params.json` (v1.5.0) gives the
  preprocessing parameters; no code is copied.
- The **IBL spike sorting pipeline** (International Brain Laboratory, doi:10.6084/m9.figshare.19705522): phase
  shift, bad-channel detection and destriping as implemented in SpikeInterface.
- **DREDge** (Windolf et al. 2025; https://github.com/evarol/dredge, MIT) as implemented in SpikeInterface.

## Not bundled

Demo data (not in the image or the repository): IBL Brain Wide Map, DANDI:000409 (CC-BY-4.0); Laquitaine et al.
ground-truth simulation, DANDI:001250 (CC-BY-4.0).

FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# The complete pinned dependency set (resolved for linux/amd64, installed without the resolver), plus
# CPU-only PyTorch for DREDge motion estimation.
COPY requirements.txt requirements-torch.txt /processor/
RUN pip install --no-deps --only-binary=:all: -r /processor/requirements.txt -r /processor/requirements-torch.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -type d -name tests -prune -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

# Lambda-safe: no cache paths here (HOME and /tmp are replaced at run time); processor.common.setup_writable_dirs
# points matplotlib / numba / triton caches at $TMPDIR at startup. USER: getpass with an arbitrary uid.
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    USER=app

# Import the stack once: the build fails if a dependency is missing.
RUN python -c "import matplotlib.pyplot, spikeinterface.full, spikeinterface.sortingcomponents.motion, pynwb, torch, mtscomp"

# Runs as uid 1000 on ECS (with a passwd entry for getpass) and as an arbitrary user on Lambda (USER=app above).
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]

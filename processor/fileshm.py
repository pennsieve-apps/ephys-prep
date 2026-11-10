"""
File-backed replacement for multiprocessing.shared_memory.SharedMemory.

SpikeInterface shares arrays between its worker processes through POSIX shared memory (/dev/shm): template
matching overlaps (circus-omp, wobble), template accumulators, shared-memory sortings. Containers get a 64 MB
/dev/shm by default (ECS tasks unless the task sets sharedMemorySize) and Lambda has none, so a Neuropixels sort
dies with SIGBUS / BrokenProcessPool. This class keeps the same interface (name, create, size, buf, close,
unlink) but maps a file in $TMPDIR/shm instead; the page cache makes it about as fast.

install() must run in the main process *and* in every worker before SpikeInterface is imported: processor.entry
calls it at module level, which spawned workers re-import.
"""
from __future__ import annotations

import mmap
import os
import secrets
import tempfile

_DIR = None


def _dir() -> str:
    global _DIR
    if _DIR is None:
        _DIR = os.path.join(os.environ.get("TMPDIR") or tempfile.gettempdir(), "shm")
        os.makedirs(_DIR, exist_ok=True)
    return _DIR


class FileSharedMemory:
    def __init__(self, name: str | None = None, create: bool = False, size: int = 0, *, track: bool = True):
        if create:
            if size <= 0:
                raise ValueError("'size' must be a positive number different from zero")
            name = name or f"si_{secrets.token_hex(8)}"
            path = os.path.join(_dir(), name)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
        else:
            if name is None:
                raise ValueError("'name' can only be None if create=True")
            path = os.path.join(_dir(), name.lstrip("/"))
            fd = os.open(path, os.O_RDWR)
            size = os.fstat(fd).st_size
        try:
            self._mmap = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self._name, self._path, self._size = name, path, size
        self._buf = memoryview(self._mmap)

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    @property
    def buf(self):
        return self._buf

    def close(self) -> None:
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def unlink(self) -> None:
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def __reduce__(self):
        return (self.__class__, (self.name, False, self.size))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, size={self.size})"


def install() -> None:
    import multiprocessing.shared_memory as msm

    msm.SharedMemory = FileSharedMemory

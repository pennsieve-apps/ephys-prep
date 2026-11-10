"""Container entry point: run the processor once and exit with its code.

The module level runs in the main process and again in every worker process SpikeInterface spawns (they re-import
the main module), so the /dev/shm-free shared memory is installed in all of them before SpikeInterface loads.
"""
import sys

from processor import fileshm

fileshm.install()

if __name__ == "__main__":     # guard: spawned worker processes import this module too
    from processor.main import main

    sys.exit(main())

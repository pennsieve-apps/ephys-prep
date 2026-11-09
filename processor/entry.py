"""Container entry point: run the processor once and exit with its code."""
import sys

if __name__ == "__main__":     # guard: SpikeInterface's worker processes re-import this module when spawned
    from processor.main import main

    sys.exit(main())

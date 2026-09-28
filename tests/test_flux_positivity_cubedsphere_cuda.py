#!/usr/bin/env python3
"""Cubed-sphere seam test on CUDA; exits 125 (skipped) without a GPU."""
import sys

import test_flux_positivity_cubedsphere

if __name__ == "__main__":
    sys.exit(test_flux_positivity_cubedsphere.main(["--device", "cuda"]))

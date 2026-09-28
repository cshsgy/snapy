#!/usr/bin/env python3
"""Moist cubed-sphere seam test on CUDA; exits 125 (skipped) without a GPU."""
import sys

import test_flux_positivity_cubedsphere_moist

if __name__ == "__main__":
    sys.exit(test_flux_positivity_cubedsphere_moist.main(["--device", "cuda"]))

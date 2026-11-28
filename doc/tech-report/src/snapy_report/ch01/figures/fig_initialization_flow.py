"""Initialize primitives, ghosts and conserved states; snapy@73c45a7c8c991209924e1137992352c335575062; no measured data."""
from ..flow import make_flow


def make_fig():
    return make_flow(['shape checks', 'physical primitive ghosts', 'primitive exchange', 'EOS and scalar density', 'solids / conserved ghosts'])

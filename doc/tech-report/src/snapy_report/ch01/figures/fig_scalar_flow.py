"""Transport passive scalars with dry mass; snapy@73c45a7c8c991209924e1137992352c335575062; no measured data."""
from ..flow import make_flow


def make_fig():
    return make_flow(['dry density / scalar density', 'ratio reconstruction', 'hydro dry flux', 'panel states / donor copy', 'scalar increment'])

"""Build options from YAML and reject unknown keys; snapy@73c45a7c8c991209924e1137992352c335575062; no measured data."""
from ..flow import make_flow


def make_fig():
    return make_flow(['YAML root', 'owning reader', 'fixed-key validation', 'sub-options', 'module construction'])

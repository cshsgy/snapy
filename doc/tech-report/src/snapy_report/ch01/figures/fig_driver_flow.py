"""Driver loop and accepted steps; snapy@73c45a7c8c991209924e1137992352c335575062; no measured data."""
from ..flow import make_flow


def make_fig():
    return make_flow(['caller: choose dt', 'all stages (chemistry in the last)', 'reject / accept', 'accepted outputs'])

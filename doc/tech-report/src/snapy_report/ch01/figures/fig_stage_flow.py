"""Accumulate and combine one stage; snapy@73c45a7c8c991209924e1137992352c335575062; no measured data."""
from ..flow import make_flow


def make_fig():
    return make_flow(['exchange (bare block)', 'publish stage / save', 'hydro increment', 'scalar / dry carry', 'user increments', 'RK / limiter / solids', 'last-stage saturation, chemistry / BC'])

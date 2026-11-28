"""Follow the hydro increment through seven sections; snapy@73c45a7c8c991209924e1137992352c335575062; no measured data."""
from ..flow import make_flow


def make_fig():
    return make_flow(['EOS', 'x1 flux / seam', 'horizontal LR / panel', 'flux / donor factors', 'divergence / geometry', 'forcing / gravity', 'implicit increment', 'x1 mass-flux sources'])

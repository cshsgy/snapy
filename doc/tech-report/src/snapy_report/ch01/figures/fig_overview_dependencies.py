"""Call ownership at snapy@73c45a7c8c991209924e1137992352c335575062; no data."""
from ..flow import make_tree


def make_fig():
    return make_tree('Mesh / MeshBlock', ['caller: drives stages', 'hydro / scalar: increments',
                                       'layout: exchanges', 'external integrator / thermo'])

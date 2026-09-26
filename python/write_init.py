#! /usr/bin/env python3

import argparse
import torch
import numpy as np
from netCDF4 import Dataset
from kintera import (
        ThermoOptions,
        ThermoX
        )
from snapy import kICY

def save_tensors(tensor_map: dict[str, torch.Tensor], filename: str):
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)
    scripted.save(filename)
    print(f'Saved tensors to {filename}')

def read_hydro(ymlfile, inpfile):
    op = ThermoOptions.from_yaml(ymlfile)
    xfrac = []

    with Dataset(inpfile, 'r') as nc:
        var = nc.variables['rho']
        rho = var[:].filled(np.nan).astype(np.float64)

        var = nc.variables['vel1']
        vel1 = var[:].filled(np.nan).astype(np.float64)

        var = nc.variables['vel2']
        vel2 = var[:].filled(np.nan).astype(np.float64)

        var = nc.variables['vel3']
        vel3 = var[:].filled(np.nan).astype(np.float64)

        var = nc.variables['press']
        pres = var[:].filled(np.nan).astype(np.float64)

        for s in op.species()[1:]:
            var = nc.variables[s]
            x = var[:].filled(np.nan).astype(np.float64)
            if (x < 0).any() or (x > 1).any():
                raise ValueError(f'Species fraction {s} out of bounds [0, 1]')
            xfrac.append(x)

    ntime, nx1, nx2, nx3 = rho.shape

    w = torch.empty((4 + len(op.species()), nx3, nx2, nx1), dtype=torch.float64)
    w[0, ...] = torch.from_numpy(rho[-1]).permute(2, 1, 0)
    w[1, ...] = torch.from_numpy(vel1[-1]).permute(2, 1, 0)
    w[2, ...] = torch.from_numpy(vel2[-1]).permute(2, 1, 0)
    w[3, ...] = torch.from_numpy(vel3[-1]).permute(2, 1, 0)
    w[4, ...] = torch.from_numpy(pres[-1]).permute(2, 1, 0)

    for i, s in enumerate(op.species()[1:]):
        w[kICY + i, ...] = torch.from_numpy(xfrac[i][-1]).permute(2, 1, 0)

    return w

if __name__ == '__main__':
    ymlfile = 'jupiter3d.yaml'
    inpfile = 'jupiter3d.out2.00051.nc'
    outfile = 'jupiter3d_init.pt'

    w = read_hydro(ymlfile, inpfile)

    data = {'hydro_w': w}
    save_tensors(data, outfile)

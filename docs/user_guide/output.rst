Output and Postprocessing
=========================

Snapy provides flexible output options for simulation results.

Output Formats
--------------

NetCDF Output
~~~~~~~~~~~~~

The default output format is NetCDF4:

.. code-block:: python

    import snapy

    options = snapy.MeshBlockOptions.from_yaml("config.yaml")
    block = snapy.MeshBlock(options)
    vars, time = block.initialize({})

    # Generate output
    block.make_outputs(vars, time)

Output files are named::

    <basename>.out<id>.<timestamp>.nc

For example::

    simulation.out1.00001.nc
    simulation.out2.00001.nc

Reading NetCDF Output
~~~~~~~~~~~~~~~~~~~~~

Read output files using netCDF4-python:

.. code-block:: python

    from netCDF4 import Dataset
    import numpy as np

    # Open output file
    nc = Dataset("simulation.out1.00001.nc", "r")

    # Read variables
    time = nc.variables["time"][:]
    density = nc.variables["rho"][:]
    velocity = nc.variables["vel1"][:]
    pressure = nc.variables["press"][:]

    # Print dimensions
    print(f"Time steps: {len(time)}")
    print(f"Density shape: {density.shape}")

    nc.close()

TorchScript Output
~~~~~~~~~~~~~~~~~~

For internal use, Snapy can save tensors in TorchScript format:

.. code-block:: python

    import torch

    # Save tensors
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule({"hydro_w": vars["hydro_w"]})
    scripted = torch.jit.script(module)
    scripted.save("output.pt")

    # Load tensors
    loaded = torch.jit.load("output.pt")
    hydro_w = loaded.hydro_w

Output Variables
----------------

Standard Variables
~~~~~~~~~~~~~~~~~~

Snapy outputs these standard hydrodynamic variables:

* ``rho``: Density (kg/m³)
* ``vel1``: Velocity in x1 direction (m/s)
* ``vel2``: Velocity in x2 direction (m/s)
* ``vel3``: Velocity in x3 direction (m/s)
* ``press``: Pressure (Pa)
* ``temp``: Temperature (K, if applicable)

Primitive Statistics
~~~~~~~~~~~~~~~~~~~~

Use ``prim_stat`` to output time-weighted hydro primitive statistics over each
output interval:

.. code-block:: yaml

    outputs:
      - type: netcdf
        variables: [prim_stat]
        dt: 300.

This writes mean and population standard deviation fields such as
``rho_mean``, ``rho_std``, ``press_mean``, ``press_std``, ``vel1_mean``,
``vel2_mean``, ``vel3_mean``, ``vel1_std``, ``vel2_std``, and ``vel3_std``.

Use ``scalar_stat`` to output time-weighted scalar primitive statistics over
the same cadence. Scalar statistic names follow the scalar primitive names, for
example ``r_tracer_a_mean`` and ``r_tracer_a_std``.

Slice Output
~~~~~~~~~~~~

Set a slice coordinate on a NetCDF output to write only the cell containing
that coordinate:

.. code-block:: yaml

    outputs:
      - type: netcdf
        variables: [prim]
        x2_slice: 0.0
        dt: 300.

Slices use lower-inclusive, upper-exclusive mesh bounds. Multiple slice
coordinates may be specified to produce line or point output. A slice
coordinate must lie inside the global mesh and cannot be combined with a sum
along the same axis.

Double-Precision Output
~~~~~~~~~~~~~~~~~~~~~~~

NetCDF output is single precision by default. Set ``double_precision`` on an
output block to write ``NC_DOUBLE`` instead:

.. code-block:: yaml

    outputs:
      - type: netcdf
        variables: [prim]
        double_precision: true
        dt: 300.

Only the ``netcdf`` writer implements it; a ``pnetcdf`` block with
``double_precision: true`` is refused at construction.

Tracer Species
~~~~~~~~~~~~~~

For simulations with tracer species:

* ``species_<name>``: Mass or mole fraction of species

Derived Quantities
~~~~~~~~~~~~~~~~~~

You can compute derived quantities in postprocessing:

.. code-block:: python

    from netCDF4 import Dataset
    import numpy as np

    nc = Dataset("simulation.out1.00001.nc", "r")

    # Read primitive variables
    rho = nc.variables["rho"][:]
    vel1 = nc.variables["vel1"][:]
    vel2 = nc.variables["vel2"][:]
    vel3 = nc.variables["vel3"][:]
    press = nc.variables["press"][:]

    # Compute kinetic energy
    ke = 0.5 * rho * (vel1**2 + vel2**2 + vel3**2)

    # Compute Mach number (gamma = 1.4)
    gamma = 1.4
    sound_speed = np.sqrt(gamma * press / rho)
    velocity_mag = np.sqrt(vel1**2 + vel2**2 + vel3**2)
    mach = velocity_mag / sound_speed

    # Compute vorticity (2D example)
    dvdx = np.gradient(vel2, axis=2)
    dudy = np.gradient(vel1, axis=1)
    vorticity = dvdx - dudy

    nc.close()

Combining Output Files
----------------------

Use pd-combine Utility
~~~~~~~~~~~~~~~~~~~~~~

Combine time series and multiple output streams:

.. code-block:: bash

    # Combine output streams 1, 2, 3
    pd-combine 1,2,3

    # Combine with custom name
    pd-combine 1,2,3 -o results

    # Combine in a specific directory
    pd-combine 1,2,3 -d output/

Python API
~~~~~~~~~~

.. code-block:: python

    from snapy.api.pd_combine import CombineTimeseries, CombineFields

    # Combine time series for a field
    stamps = ["00001", "00002", "00003", "00004", "00005"]
    CombineTimeseries("simulation", "out1", stamps, path="./output")

    # Combine multiple fields
    CombineFields("simulation", "1,2,3", "combined", path="./output")

Inspecting Output Files
-----------------------

Use pd-inspect Utility
~~~~~~~~~~~~~~~~~~~~~~

Inspect TorchScript .pt files:

.. code-block:: bash

    # Inspect a single file
    pd-inspect output.pt

    # Inspect all files in a tar archive
    pd-inspect archive.tar.gz

    # Inspect multiple files
    pd-inspect output1.pt output2.pt output3.pt

Python API
~~~~~~~~~~

.. code-block:: python

    from snapy.api.pd_inspect import inspect_pt_file

    # Inspect file
    inspect_pt_file("output.pt")

Visualization
-------------

Using Matplotlib
~~~~~~~~~~~~~~~~

Basic 2D visualization:

.. code-block:: python

    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    import numpy as np

    # Read data
    nc = Dataset("simulation.out1.00001.nc", "r")
    time = nc.variables["time"][:]
    density = nc.variables["rho"][:]
    nc.close()

    # Plot snapshot at last time step
    plt.figure(figsize=(10, 8))
    plt.imshow(density[-1, :, :, 0], origin="lower", aspect="auto")
    plt.colorbar(label="Density (kg/m³)")
    plt.xlabel("x2")
    plt.ylabel("x3")
    plt.title(f"Density at t = {time[-1]:.3f} s")
    plt.savefig("density_snapshot.png", dpi=150)

Time Series Animation
~~~~~~~~~~~~~~~~~~~~~

Create animation of time evolution:

.. code-block:: python

    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    # Read data
    nc = Dataset("simulation.out1.nc", "r")
    time = nc.variables["time"][:]
    density = nc.variables["rho"][:]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initial frame
    im = ax.imshow(density[0, :, :, 0], origin="lower", aspect="auto",
                   vmin=density.min(), vmax=density.max())
    plt.colorbar(im, ax=ax, label="Density (kg/m³)")
    title = ax.set_title(f"t = {time[0]:.3f} s")

    def update(frame):
        im.set_data(density[frame, :, :, 0])
        title.set_text(f"t = {time[frame]:.3f} s")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(time),
                                  interval=50, blit=True)
    anim.save("density_evolution.mp4", fps=20, dpi=150)

    nc.close()

Using ParaView
~~~~~~~~~~~~~~

For 3D visualization, use ParaView:

1. Open NetCDF file in ParaView
2. Select the appropriate reader (NetCDF files generic reader)
3. Apply filters and visualizations as needed

Export formats compatible with ParaView:

.. code-block:: yaml

    output:
      format: netcdf
      variables: [density, velocity, pressure]

Data Analysis
-------------

Time-Averaged Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from netCDF4 import Dataset
    import numpy as np

    nc = Dataset("simulation.out1.nc", "r")
    density = nc.variables["rho"][:]

    # Time-averaged density
    rho_mean = np.mean(density, axis=0)

    # RMS fluctuations
    rho_fluct = density - rho_mean[np.newaxis, :, :, :]
    rho_rms = np.sqrt(np.mean(rho_fluct**2, axis=0))

    nc.close()

Spatial Statistics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Horizontal average
    rho_horz_mean = np.mean(density, axis=(2, 3))

    # Vertical profile
    rho_vert_profile = np.mean(density, axis=(0, 2, 3))

Power Spectra
~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from scipy import fft

    # 2D power spectrum
    rho_2d = density[-1, :, :, 0]

    # FFT
    rho_fft = fft.fft2(rho_2d)
    power = np.abs(rho_fft)**2

    # Radially averaged spectrum
    ny, nx = rho_2d.shape
    ky = fft.fftfreq(ny)
    kx = fft.fftfreq(nx)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)

    # Bin by wavenumber
    k_bins = np.linspace(0, k.max(), 50)
    power_spectrum = np.zeros(len(k_bins) - 1)

    for i in range(len(k_bins) - 1):
        mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        power_spectrum[i] = np.mean(power[mask])

Custom Output Functions
-----------------------

Register custom output:

.. code-block:: python

    import snapy
    import torch

    def custom_output(vars):
        """Return custom derived quantities."""
        # Extract variables
        rho = vars["hydro_w"][snapy.kIDN, :, :, :]
        vel1 = vars["hydro_w"][snapy.kIV1, :, :, :]
        vel2 = vars["hydro_w"][snapy.kIV2, :, :, :]
        vel3 = vars["hydro_w"][snapy.kIV3, :, :, :]
        press = vars["hydro_w"][snapy.kIPR, :, :, :]

        # Compute derived quantities
        ke = 0.5 * rho * (vel1**2 + vel2**2 + vel3**2)
        gamma = 1.4
        ie = press / (gamma - 1.0)
        total_energy = ke + ie

        return {
            "kinetic_energy": ke,
            "internal_energy": ie,
            "total_energy": total_energy,
        }

    # Register callback
    block.set_user_output_func(custom_output)

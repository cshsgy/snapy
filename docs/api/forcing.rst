Forcing Module
==============

.. module:: snapy

Classes
-------

ConstGravityOptions
~~~~~~~~~~~~~~~~~~~

.. class:: ConstGravityOptions

   Constant gravity forcing configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> ConstGravityOptions

      Load ConstGravityOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: ConstGravityOptions loaded from file
      :rtype: ConstGravityOptions

   .. method:: g() -> float
               g(value: float) -> ConstGravityOptions

      Get or set the gravitational acceleration.

      :return: Gravitational acceleration
      :rtype: float

CoriolisOptions
~~~~~~~~~~~~~~~

.. class:: CoriolisOptions

   Coriolis forcing configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> CoriolisOptions

      Load CoriolisOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: CoriolisOptions loaded from file
      :rtype: CoriolisOptions

   .. method:: omega() -> float
               omega(value: float) -> CoriolisOptions

      Get or set the rotation rate.

      :return: Rotation rate
      :rtype: float

DiffusionOptions
~~~~~~~~~~~~~~~~

.. class:: DiffusionOptions

   Isotropic hydro diffusion configuration for Cartesian meshes.

   .. method:: nu_iso() -> float
               nu_iso(value: float) -> DiffusionOptions

      Get or set the viscosity: kinematic by default, or the dynamic
      viscosity ``mu`` when :py:meth:`dynamic` is true.

   .. method:: kappa_iso() -> float
               kappa_iso(value: float) -> DiffusionOptions

      Get or set the thermal coefficient: a diffusivity in units of length
      squared per time by default, or the conductivity ``k`` when
      :py:meth:`dynamic` is true. In the default (kinematic) form the
      conductive energy flux is ``-rho * cv * kappa_iso * grad(T)``, where
      ``cv`` is the local equation-of-state mixture specific heat at constant
      volume; in the dynamic form it is ``-k * grad(T)``, with no face density.

   .. method:: dynamic() -> bool
               dynamic(value: bool) -> DiffusionOptions

      Read ``nu_iso`` as ``mu`` and ``kappa_iso`` as ``k``. Default ``False``,
      which leaves the previous path bit-identical.

   .. method:: nu_scale_x1_table() -> torch.Tensor
               nu_scale_x1_table(value: torch.Tensor) -> DiffusionOptions
               kappa_scale_x1_table() -> torch.Tensor
               kappa_scale_x1_table(value: torch.Tensor) -> DiffusionOptions

      An x1 profile multiplying the kinematic ``nu_iso`` (or ``kappa_iso``):
      a ``(2, n)`` float64 tensor of x1 knots (strictly increasing) and
      scale values (finite, strictly positive), ``n >= 2``. It is linear
      between knots, held beyond the ends, and interpolated onto each
      block's x1 cell centres, so it works with any number of blocks along
      x1. Undefined by default (a uniform coefficient). The time-step bound
      uses the largest scale.

   .. method:: nu_scale_x1() -> torch.Tensor
               nu_scale_x1(value: torch.Tensor) -> DiffusionOptions
               kappa_scale_x1() -> torch.Tensor
               kappa_scale_x1(value: torch.Tensor) -> DiffusionOptions

      The same profile given per cell: a 1-D float64 tensor with one value
      per x1 cell centre of the block, ghosts included (``nc1`` values).
      Refused when x1 is split into more than one block, since every block
      shares these options; use the table there.

   A profile is refused together with ``dynamic=True``, when both the table
   and the per-cell form are given, and when it is set, replaced or changed,
   or ``dynamic`` is set to true beside it, after the MeshBlock is built (at
   the next ``forward`` or ``max_time_step``, which compare the profile with a
   copy taken at build, so a write through a NumPy array behind
   ``torch.from_numpy`` or through ``.data`` is caught too). An inference tensor (made under
   ``torch.inference_mode()``) is refused, since its in-place changes cannot be
   detected; pass an ordinary tensor.

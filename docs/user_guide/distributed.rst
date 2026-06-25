Distributed Simulations
=======================

Snapy supports distributed parallel simulations using UCX on Linux. Gloo
remains the CPU-only and macOS distributed backend.

Distributed Architecture
------------------------

Snapy uses domain decomposition to split the computational domain across multiple processes. Each process owns a subdomain and communicates with neighbors to exchange boundary (halo) data.

Setup
-----

Linux builds use the installed ``commux`` package for UCX communication. Select
UCX as the backend without specifying a communication device:

.. code-block:: yaml

    distribute:
      backend: ucx

UCX is the default backend on builds configured with UCX, so the ``backend``
entry may be omitted. Set ``backend: gloo`` or ``BACKEND=gloo`` to opt into
Gloo.

Communication automatically follows each tensor's device type. For CUDA
execution, build with ``-DCUDA=ON`` and set ``DEVICE=cuda`` when launching the
provided examples. ``BACKEND`` overrides the configured transport.

Basic distributed setup using ``torchrun``:

.. code-block:: bash

    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 simulation.py

    # Multi-node (2 nodes, 4 GPUs each)
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr="master_node" --master_port=29500 \
             simulation.py

Python Code
-----------

Basic distributed simulation:

.. code-block:: python

    import torch.distributed as dist
    from snapy import exchange
    import snapy
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--layout", default="slab",
                       choices=["slab", "cubed", "cubed_sphere"])
    parser.add_argument("--px3", type=int, default=2)
    parser.add_argument("--px2", type=int, default=2)
    parser.add_argument("--px1", type=int, default=1)
    args = parser.parse_args()

    # Initialize distributed environment
    layout, ranks, device, info = exchange.init_dist(
        args,
        periodic_x3=True,
        periodic_x2=True
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Load configuration
    options = snapy.MeshBlockOptions.from_yaml("config.yaml")
    block = snapy.MeshBlock(options)

    # Initialize variables
    vars, time = block.initialize({})

    # Initialize communication buffers
    send_bufs, recv_bufs = exchange.init_buffers_2d(
        layout, rank, block, vars
    )

    # Main simulation loop
    while time < max_time:
        dt = block.max_time_step(vars)

        # Exchange halo data
        exchange.slab_exchange(block, vars, ranks, send_bufs, recv_bufs)

        # Forward integration
        vars = block.forward(dt, 0, vars)
        time += dt

        # Output (rank 0 only)
        if rank == 0 and time >= next_output:
            block.make_outputs(vars, time)
            next_output += output_interval

    # Cleanup
    dist.destroy_process_group()

Domain Layouts
--------------

Slab Layout
~~~~~~~~~~~

2D decomposition in the x2-x3 plane:

.. code-block:: python

    from snapy import SlabLayout

    # Create slab layout: 4x2 = 8 processes
    layout = SlabLayout(
        nb3=4,  # 4 blocks in x3 direction
        nb2=2,  # 2 blocks in x2 direction
        periodic_x3=True,
        periodic_x2=True
    )

    # Get location of rank 5
    x3, x2 = layout.loc_of(5)  # Returns (1, 1)

    # Get neighbor rank
    neighbor = layout.neighbor_rank(x3, x2, dx3=1, dx2=0)

Cubed Layout
~~~~~~~~~~~~

3D decomposition:

.. code-block:: python

    from snapy import CubedLayout

    # Create cubed layout: 4x2x2 = 16 processes
    layout = CubedLayout(
        nb3=4,  # 4 blocks in x3 direction
        nb2=2,  # 2 blocks in x2 direction
        nb1=2,  # 2 blocks in x1 direction
        periodic_x3=True,
        periodic_x2=True,
        periodic_x1=False
    )

    # Get location
    x3, x2, x1 = layout.loc_of(rank)

    # Get neighbor
    neighbor = layout.neighbor_rank(x3, x2, x1, dx3=1, dx2=0, dx1=0)

Cubed Sphere Layout
~~~~~~~~~~~~~~~~~~~

For global atmospheric simulations:

.. code-block:: python

    from snapy import CubedSphereLayout

    # Create cubed sphere layout: 6 faces × 4×4 = 96 processes
    layout = CubedSphereLayout(nb_per_face=4)

    # Get location (face, x3, x2)
    face, x3, x2 = layout.loc_of(rank)

    # Get neighbor
    neighbor = layout.neighbor_rank(face, x3, x2, dx3=1, dx2=0)

Communication Patterns
----------------------

Manual Exchange
~~~~~~~~~~~~~~~

For custom communication patterns:

.. code-block:: python

    import torch.distributed as dist

    # Send to neighbor
    if right_rank != -1:
        req_send = dist.isend(send_buffer, right_rank)
        req_recv = dist.irecv(recv_buffer, right_rank)
        req_send.wait()
        req_recv.wait()

Collective Operations
~~~~~~~~~~~~~~~~~~~~~

Use PyTorch's collective operations:

.. code-block:: python

    import torch.distributed as dist

    # All-reduce (sum across all processes)
    total_energy = local_energy.clone()
    dist.all_reduce(total_energy, op=dist.ReduceOp.SUM)

    # Broadcast from rank 0
    if rank == 0:
        config_tensor = torch.tensor([dt, time])
    else:
        config_tensor = torch.zeros(2)
    dist.broadcast(config_tensor, src=0)

Performance Optimization
------------------------

Communication Overlap
~~~~~~~~~~~~~~~~~~~~~

Overlap communication with computation:

.. code-block:: python

    # Start non-blocking communication
    exchange.slab_exchange(block, vars, ranks, send_bufs, recv_bufs)

    # Compute interior cells while communication is in progress
    # (requires manual implementation of interior-only update)

    # Wait for communication to complete
    # (handled internally by slab_exchange)

GPU-Direct Communication
~~~~~~~~~~~~~~~~~~~~~~~~

For UCX with GPU-direct support, select CUDA memory and configure the desired
UCX transports for the cluster:

.. code-block:: bash

    export BACKEND=ucx
    export DEVICE=cuda
    export UCX_TLS=rc,cuda_copy,cuda_ipc

Load Balancing
--------------

The domain decomposition should balance work across processes. For uniform grids, equal subdivision works well. For non-uniform grids or adaptive refinement, consider dynamic load balancing.

Slurm Integration
-----------------

Example Slurm script:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=snapy_sim
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-node=4
    #SBATCH --time=24:00:00

    # Load modules
    module load cuda/12.1
    export BACKEND=ucx
    export DEVICE=cuda

    # Run simulation
    srun torchrun --nnodes=$SLURM_NNODES \
                  --nproc_per_node=4 \
                  --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
                  --master_port=29500 \
                  simulation.py --device=cuda --layout=slab --px3=4 --px2=4

Debugging
---------

Debug distributed simulations:

.. code-block:: python

    import torch.distributed as dist

    rank = dist.get_rank()

    # Print from all ranks
    print(f"[Rank {rank}] Initialized successfully")

    # Synchronize all processes
    dist.barrier()

    # Check for NaN values
    if torch.isnan(vars["hydro_w"]).any():
        print(f"[Rank {rank}] NaN detected!")
        dist.barrier()
        raise RuntimeError("NaN detected")

Enable UCX debugging:

.. code-block:: bash

    export UCX_LOG_LEVEL=info

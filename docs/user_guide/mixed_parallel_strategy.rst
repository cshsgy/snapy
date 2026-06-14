Mixed Parallel Strategy
=======================

This note describes the mixed parallel execution strategy used by ``snapy``
after the multi-block communication refactor. "Mixed parallel" means that the
global domain may be split in two dimensions at the same time:

* across multiple processes, using UCX or Gloo for inter-process exchange
* across multiple ``MeshBlock`` objects inside a single process

The design goal is to let those two forms of parallelism compose without
duplicating communication logic or letting physics modules bypass the transport
layer. The result is a layered model in which ``Mesh`` schedules work,
``MeshBlock`` owns state validity, and ``Layout`` owns topology and transport
mapping.

Overview
--------

At runtime, a single process may own more than one block. That makes the system
different from a pure one-rank-one-block decomposition. A boundary exchange can
therefore fall into three cases:

* local-only: both neighboring blocks belong to the same process
* remote-only: the neighbor lives on another process
* mixed: a process owns some local neighbors and also exchanges with remote
  peers

Figure 1: Mixed decomposition model

.. code-block:: text

   Global domain
   +---------------------------+---------------------------+
   | Process 0                 | Process 1                 |
   |                           |                           |
   |  Block 0   <local> Block 1|  Block 3   <local> Block 4|
   |     |                     |     |                     |
   |   remote                remote  |                     |
   |     v                     ^     v                     |
   |  Block 2   <local> Block 5|  Block 6   <local> Block 7|
   |                           |                           |
   +---------------------------+---------------------------+

   Legend:
   - horizontal arrows inside a process are local block-to-block exchange
   - vertical arrows crossing the process boundary are remote exchange

UCX and Gloo both provide tagged point-to-point messaging, allowing each block
to launch remote exchanges independently while matching messages by tag.

Why A Mixed Strategy Is Needed
------------------------------

The old split implementation had three owners for communication:

* ``Mesh`` handled some multi-block halo exchange
* ``MeshBlock`` handled some direct exchange
* ``Hydro`` directly triggered cubed-sphere LR-state communication

That arrangement made it easy for one path to remain correct while another
quietly diverged. The cubed-sphere single-process multi-block failure was the
clearest symptom: cartesian exchange remained correct, but cubed-sphere
LR-state exchange broke because the transport path was split across layers.

The current design removes that duplication. Every exchanged state follows the
same ownership model:

* ``Mesh`` starts asynchronous block jobs
* ``MeshBlock`` decides when a block's state must be exchanged
* ``Layout`` serializes, launches, and finalizes the exchange
* physics modules prepare payloads, but do not own transport

Figure 2: Ownership stack

.. code-block:: text

   +--------------------------------------------------------------+
   | Mesh                                                         |
   |  - creates blocks                                            |
   |  - launches async jobs                                       |
   |  - performs global reductions and output aggregation         |
   +------------------------------+-------------------------------+
                                  |
                                  v
   +--------------------------------------------------------------+
   | MeshBlock                                                    |
   |  - owns begin/launch/finalize exchange lifecycle             |
   |  - ensures local state is valid before returning             |
   +------------------------------+-------------------------------+
                                  |
                                  v
   +--------------------------------------------------------------+
   | Layout                                                       |
   |  - neighbor topology                                         |
   |  - serialization / deserialization                           |
   |  - local rendezvous and remote process-group work            |
   +------------------------------+-------------------------------+
                                  |
                                  v
   +--------------------------------------------------------------+
   | Snapy communication facade                                   |
   |  - native UCX or legacy c10d transport                        |
   +--------------------------------------------------------------+

Core Data Model
---------------

Each process owns ``blocks_per_process`` local blocks. The layout options encode
two coordinate systems:

* process coordinates: which process owns a region of the global partition
* local-block coordinates: which block inside that process owns a local shard

This leads to two useful rank mappings:

* ``process_rank``: rank in the distributed process group
* ``block_rank``: logical rank in the full partitioned mesh

The logical block rank is what topology calculations operate on. The process
rank is what remote communication uses. The bridge between them is the layout's
mapping helpers:

* ``owner_process_rank(block_rank)``
* ``local_block_index(block_rank)``
* ``global_block_rank(process_rank, local_block)``

These mappings let a single transport API cover both local and remote
neighbors. A caller can ask for the neighbor of a logical block, then decide
whether the transport is a same-process copy or a remote process-group
operation.

Execution Model
---------------

``Mesh`` is intentionally a scheduler, not an exchanger. It creates the local
blocks and launches them asynchronously. The design assumes that a block should
return from its stage work only after its communicated state is valid.

Flowchart 1: Per-stage execution

.. code-block:: text

   +-------------------+
   | Mesh::forward()   |
   +-------------------+
             |
             v
   +---------------------------+
   | launch block jobs async   |
   +---------------------------+
             |
             v
   +---------------------------+
   | MeshBlock::forward()      |
   | 1. local physics update   |
   | 2. BC / solid handling    |
   | 3. begin_exchange()       |
   | 4. launch_exchange()      |
   | 5. finalize_exchange()    |
   +---------------------------+
             |
             v
   +---------------------------+
   | return block with valid   |
   | ghost/interface state     |
   +---------------------------+
             |
             v
   +-------------------+
   | Mesh joins jobs   |
   +-------------------+

This scheduling model preserves a single responsibility boundary:
``Mesh`` decides *when* blocks run, while ``MeshBlock`` decides *what exchange
must happen before the block is considered ready*.

Exchange Lifecycle
------------------

The block-owned exchange API is split into three phases because not every
transport can be treated as a one-shot send/receive.

1. ``begin_exchange(vars, opts)``

   The layout serializes outgoing faces into block-owned send buffers. At this
   point the payload exists, but transport has not completed.

2. ``launch_exchange(opts, works)``

   Local neighbors rendezvous with sibling blocks on the same process, while
   remote neighbors launch process-group work. The launch phase creates opaque
   ``work`` handles for remote completion.

3. ``finalize_exchange(vars, opts, works)``

   The block waits for remote work, deserializes receive buffers, and applies
   any topology-specific post-processing.

Flowchart 2: Exchange decision tree

.. code-block:: text

   +-----------------------------+
   | For each logical neighbor   |
   +-----------------------------+
                |
                v
       +----------------------+
       | same process owner?  |
       +----------------------+
         | yes                    | no
         v                        v
   +------------------+    +----------------------+
   | local rendezvous |    | launch remote work   |
   | register sender  |    | post recv/send in    |
   | copy to recv buf |    | backend-safe order    |
   +------------------+    +----------------------+
         \                        /
          \                      /
           v                    v
          +----------------------+
          | finalize deserialize |
          +----------------------+

This API keeps the simple caller path intact as well: ``exchange(vars, opts)``
remains a wrapper that performs begin, launch, and finalize in sequence.

Local And Remote Paths
----------------------

Local exchange is not treated as a special high-level shortcut in ``Mesh``.
Instead, it is implemented inside ``Layout`` so that the same serialization and
deserialization rules apply to both transport modes.

Local path
~~~~~~~~~~

For local neighbors, the layout registers each participating block in a
per-process rendezvous table keyed by exchange configuration. Once all local
participants for that exchange are present, the leader performs direct buffer
copies between the matching send and receive buffers.

This avoids two problems:

* a redundant mesh-level implementation of neighbor mapping
* backend-specific behavior leaking into local-only exchange

Remote path
~~~~~~~~~~~

For remote neighbors, the layout launches process-group operations. The layout
already knows the sender/receiver offsets, buffer shapes, and topology remap
rules, so it is the right layer to decide ordering.

The design rule is simple: local and remote paths must produce the same receive
buffers for the same logical exchange.

Cubed-Sphere LR-State Exchange
------------------------------

The cubed-sphere case adds another requirement: reconstructed LR interface
states in ``DIM2`` and ``DIM3`` cannot be treated as completely independent
one-shot exchanges. The face remap is topology-sensitive and the staged order
must be preserved.

The current design keeps that staged order but still routes it through the
block-owned exchange interface:

1. Hydro reconstructs LR state tensors
2. ``MeshBlock`` begins the ``DIM2`` and ``DIM3`` exchanges
3. ``Layout`` launches local and remote work using topology-aware remap rules
4. ``MeshBlock`` finalizes in the required order
5. Hydro continues with flux evaluation

That preserves the correct ownership:

* Hydro owns reconstruction math
* ``MeshBlock`` owns exchange lifecycle
* ``Layout`` owns cubed-sphere remap and transport

Correctness Invariants
----------------------

Any future change to the mixed parallel path should preserve these invariants:

* ``Mesh`` never reimplements exchange topology.
* ``MeshBlock`` is the only owner of exchange orchestration.
* ``Layout`` is the only owner of neighbor mapping and remap logic.
* local-neighbor and remote-neighbor paths produce identical receive buffers.
* tagged remote exchanges use stable source, destination, and physics tags.
* cubed-sphere-specific behavior stays in cubed-sphere layout logic, not in
  generic mesh scheduling.

Debugging Strategy
------------------

Mixed parallel bugs become easier to diagnose when the failure is assigned to
the correct layer.

* If the wrong neighbor is selected, inspect ``Layout`` topology.
* If the correct payload is sent in the wrong order, inspect layout launch
  ordering and process-wide grouping.
* If the wrong tensor is being exchanged, inspect the caller that prepared the
  ``Variables`` bundle.
* If a block returns before its state is valid, inspect ``MeshBlock`` stage and
  finalize sequencing.

A practical debugging ladder is:

* local exchange regression
* mixed local+remote exchange regression
* cartesian decomposition example
* cubed-sphere decomposition example

The most useful comparison is often between a same-process multi-block case and
an equivalent multi-process case. The remote path is usually easier to reason
about, and the local path should reproduce its receive buffers exactly.

Summary
-------

The mixed parallel strategy in ``snapy`` is based on one central principle:
composition should happen through a single exchange contract, not through
multiple ad hoc implementations.

In practice that means:

* ``Mesh`` launches work
* ``MeshBlock`` owns exchange lifecycle
* ``Layout`` owns topology and transport mapping
* backend-specific ordering stays inside layout transport logic

That separation is what makes single-process multi-block, multi-process
single-block, and mixed decompositions behave like the same algorithm instead
of three different communication systems.

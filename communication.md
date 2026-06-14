# Communication Structure

This note describes how communication is organized in `snapy` after the
multi-block refactor. The main design goal is to keep communication ownership
close to the data that is being exchanged, while leaving high-level scheduling
at the mesh level.

## Design goals

1. `Mesh` should coordinate work, not implement exchange logic.
2. `MeshBlock` should own the communication needed to keep its state valid.
3. `Layout` should remain the source of truth for neighbor topology, message
   shape, serialization, and deserialization.
4. Hydro-specific communication, including cubed-sphere LR-state exchange,
   should use the same block-owned communication path instead of bypassing it.
5. The same code path should work for:
   - single block
   - multiple processes
   - multiple blocks on one process
   - mixed process/block decompositions

## Main objects and responsibilities

### `Mesh`

`Mesh` owns the collection of blocks and launches block work asynchronously.
Its job is orchestration:

- create blocks and shared metadata
- launch initialization and stage advancement on each block
- collect global reductions such as timestep and redo decisions
- aggregate outputs and finalize the run

`Mesh` should not decide how ghost zones or interface states move between
neighbors. That logic used to live partly in `Mesh`, which created duplicate
ownership and made it easy for one communication path to diverge from another.

### `MeshBlock`

`MeshBlock` is the owner of block-local state validity. If a block updates or
reconstructs state that needs neighbor information, the block is also
responsible for triggering the exchange that makes that state usable.

The core block-owned exchange interface is:

- `begin_exchange(vars, opts)`
- `launch_exchange(opts, works)`
- `finalize_exchange(vars, opts, works)`
- `exchange(vars, opts)` as the one-shot wrapper

This split exists because some communication patterns need a staged flow rather
than a single immediate send/receive/deserialize sequence.

### `Layout`

`Layout` is the communication policy and topology layer. It knows:

- which neighbors exist
- whether a neighbor is local or remote
- how an exchange is serialized
- how received data is deserialized back into tensors
- how to remap offsets for a particular topology

`Layout` does not own simulation advancement. It provides the transport and
neighbor-mapping rules that `MeshBlock` uses.

### `Hydro`

`Hydro` owns reconstruction and flux calculations. It should not directly own
cross-block communication. When Hydro needs exchanged interface states, it
builds the exchange payload and routes it through the `MeshBlock` communication
path.

This is especially important for cubed-sphere LR-state exchange, where the
communication ordering matters.

## Why communication ownership moved out of `Mesh`

The earlier design split communication across multiple layers:

- `Mesh` exchanged regular state for multi-block runs
- `MeshBlock` exchanged state in single-block contexts
- `Hydro` directly invoked layout communication for LR states

That arrangement had two problems.

First, the same conceptual operation existed in more than one place. This made
it difficult to guarantee that all paths handled local neighbors, remote
neighbors, and topology-specific remapping in the same way.

Second, the cubed-sphere `1 proc x N blocks` case exposed a gap: the cartesian
path happened to work, but cubed-sphere LR-state exchange diverged because one
communication path did not preserve the same staged semantics as the other.

Moving exchange ownership into `MeshBlock` fixes the layering:

- `Mesh` schedules
- `MeshBlock` requests communication
- `Layout` performs communication policy
- `Hydro` prepares payloads but does not drive transport directly

## Exchange lifecycle

For most exchanged state, the lifecycle is:

1. `MeshBlock` prepares a `Variables` bundle containing the fields to exchange.
2. `begin_exchange()` serializes the outgoing faces through the layout.
3. `launch_exchange()` prepares local copies and launches remote process-group
   work.
4. `finalize_exchange()` waits for remote work, deserializes received data, and
   performs any post-processing such as corner fills.

This staged interface lets the caller overlap preparation and transport when
needed, but it still supports a simple one-shot `exchange()` call for common
cases.

## Local vs remote neighbors

The layout distinguishes two classes of neighbors.

### Remote neighbors

For remote neighbors, communication is done through the configured transport.
The layout serializes send buffers, launches tagged point-to-point work, then
deserializes the received buffers during finalize. Linux builds provide a
native UCX transport; Gloo remains available for CPU-only and macOS builds.

### Local neighbors

For neighbors that live on the same process, going through the backend is both
unnecessary and fragile. The layout therefore performs a process-local
rendezvous:

- local layouts register themselves by process rank and local block index
- a per-process exchange state gathers all local participants for a given
  exchange configuration
- once all participating local blocks have prepared their send buffers, the
  leader copies local neighbor data directly into the matching receive buffers
- all local participants are released and finalize normally

This keeps local and remote exchanges under the same `MeshBlock -> Layout`
contract while avoiding duplicate transport logic in `Mesh`.

## Cubed-sphere LR-state exchange

Cubed-sphere LR-state exchange is the most sensitive case.

Hydro reconstructs left/right interface states separately for `DIM2` and `DIM3`.
Those reconstructed tensors are not just another ghost-zone update. Across cube
faces, the mapping between sender and receiver depends on the topology, and the
old implementation relied on a staged ordering:

1. prepare `DIM2`
2. prepare `DIM3`
3. launch transport
4. finalize `DIM2`
5. finalize `DIM3`

Treating `DIM2` and `DIM3` as independent one-shot exchanges is incorrect for
this case. The current design therefore keeps the staged ordering, but routes
it through `MeshBlock` rather than letting Hydro talk to `Layout` directly.

This preserves two important invariants:

- Hydro owns reconstruction math
- `MeshBlock` and `Layout` own communication

## Initialization and stage advancement

Initialization and stage advancement follow the same ownership rule.

### Initialization

`Mesh` launches block initialization asynchronously. Each block performs:

1. local initialization
2. primitive exchange
3. scalar exchange if present
4. final block initialization steps

The block returns only after its initialized state is communication-consistent.

### Stage advancement

During each stage, `Mesh` launches block work asynchronously. Each block:

1. advances local physics
2. applies any required boundary or solid handling
3. exchanges the updated state through the block-owned path
4. returns with state ready for the next stage

This keeps stage correctness local to the block implementation and prevents the
mesh scheduler from needing to understand field-specific communication details.

## Invariants to preserve

Any future communication work should preserve these invariants:

1. There is one owner for transport orchestration: `MeshBlock`.
2. There is one owner for topology and buffer mapping: `Layout`.
3. `Mesh` never reimplements neighbor exchange rules.
4. Physics modules may construct exchange payloads, but they should not own
   backend communication directly.
5. Local-neighbor and remote-neighbor paths must produce identical receive
   buffers.
6. Cubed-sphere-specific remapping stays in the cubed-sphere layout, not in
   generic mesh scheduling code.
7. Callers use the Snapy communication facade rather than accessing a
   torch.distributed process group directly.

## Practical debugging guidance

When debugging communication, first identify which layer owns the failure:

- If neighbors or face mappings are wrong, inspect `Layout`.
- If serialized fields or exchange ordering are wrong, inspect `MeshBlock`.
- If the wrong tensors are being prepared for exchange, inspect the physics
  module such as `Hydro`.
- If work is not being launched or aggregated correctly, inspect `Mesh`.

For multi-block single-process bugs, compare local-neighbor behavior against the
equivalent multi-process case. The remote path is often easier to reason about,
and the local path should produce the same receive buffers.

## Summary

The communication structure is intentionally layered:

- `Mesh` schedules work
- `MeshBlock` owns exchange lifecycle
- `Layout` owns topology and transport mapping
- physics modules build payloads but do not own communication orchestration

That separation removes duplicated exchange logic, makes local and remote
neighbors follow the same contract, and keeps cubed-sphere-specific complexity
inside the layout and staged block-owned exchange flow.

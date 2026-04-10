import os
import torch
import snapy
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
import numpy as np
from typing import List, Optional

@torch.compile
def get_buffer_id(dx: int, dy: int, dz: int=0):
    return dx % 3 + (dy % 3) * 3 + (dz % 3) * 9

def init_dist(args,
              periodic_x1: bool=False,
              periodic_x2: bool=False,
              periodic_x3: bool=False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1:
        if args.device == "cpu":
            dist.init_process_group(backend="gloo", init_method="env://")
        else:
            dist.init_process_group(backend="nccl", init_method="env://")
        snapy.distributed.set_process_group(dist_c10d._get_default_group())
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    if args.device == "cuda":
        ngpu = torch.cuda.device_count()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, ngpu)))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    px, py, pz = args.px3, args.px2, args.px1

    if args.layout == "slab":
        assert pz == 1, "px1 must be 1 for slab layout"
        assert px * py == world_size, f"px2*px3 ({px}*{py}) != world_size ({world_size})"
        layout = snapy.SlabLayout(px, py, periodic_x3, periodic_x2)
        loc = layout.loc_of(rank)

        info = snapy.DistributeInfo()
        info.nb3(px)
        info.nb2(py)
        info.lx3(loc[0])
        info.lx2(loc[1])
    elif args.layout == "cubed":
        assert px * py * pz == world_size, f"px1*px2*px3 ({px}*{py}*{pz}) != world_size ({world_size})"
        layout = snapy.CubedLayout(px, py, pz, periodic_x3, periodic_x2, periodic_x1)
        loc = layout.loc_of(rank)

        info = snapy.DistributeInfo()
        info.nb3(px)
        info.nb2(py)
        info.nb1(pz)
        info.lx3(loc[0])
        info.lx2(loc[1])
        info.lx1(loc[2])
    else: # cubed_sphere
        assert pz == 1, "px1 must be 1 for cubed_sphere layout"
        assert px == py, "px2 must equal px3 for cubed_sphere layout"
        assert 6 * px * py == world_size, f"6*px2*px3 ({px}*{py}) != world_size ({6*world_size})"
        layout = snapy.CubedSphereLayout(px)
        loc = layout.loc_of(rank)

        info = snapy.DistributeInfo()
        info.nb3(px)
        inof.nb2(py)
        info.face(loc[0])
        info.lx3(loc[1])
        info.lx2(loc[2])

    info.gid(rank)
    if args.layout == "cubed":  # 3D decomposition
        ranks = np.zeros(27, dtype=int)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx != 0 or dy != 0 or dz != 0:
                        offset = (dx, dy, dz)
                        bid = get_buffer_3d(*offset)
                        ranks[bid] = layout.neighbor_rank(*layout.loc_of(rank), *offset)
        # my rank
        ranks[get_buffer_3d(0, 0, 0)] = rank
    else:  # 2D decomposition
        ranks = np.zeros(9, dtype=int)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    offset = (dx, dy, 0)
                    bid = get_buffer_id(*offset)
                    ranks[bid] = layout.neighbor_rank(*layout.loc_of(rank), *offset)
        # my rank
        ranks[get_buffer_id(0, 0, 0)] = rank

    return layout, ranks, device, info

def init_buffers_2d(layout, rank,
                    block: snapy.MeshBlock,
                    block_vars: dict[str, torch.Tensor]):
    send_bufs = [None] * 9
    recv_bufs = [None] * 9

    for x3_offset in [-1, 0, 1]:
        for x2_offset in [-1, 0, 1]:
            if x3_offset == 0 and x2_offset == 0: continue # skip self
            offset = (x3_offset, x2_offset, 0)
            loc = layout.loc_of(rank)
            nb = layout.neighbor_rank(*loc, *offset)
            if nb == -1: continue # no neighbor

            # invalidate block neighbor
            block.options.set_bfunc(*offset, None)

            bid = get_buffer_id(*offset)
            part = block.part(offset)
            nhydro, *dims = block_vars["hydro_u"][part].shape
            send_bufs[bid] = torch.empty((nhydro, *dims),
                                         device=block_vars["hydro_u"].device,
                                         dtype=block_vars["hydro_u"].dtype)
            recv_bufs[bid] = torch.empty_like(send_bufs[bid])
    return send_bufs, recv_bufs

@torch.compile
def serialize_2d(block: snapy.MeshBlock,
                 block_vars: dict[str, torch.Tensor],
                 send_bufs: List[Optional[torch.Tensor]]):
    nhydro = block_vars["hydro_u"].shape[0]

    for x3_offset in [-1, 0, 1]:
        for x2_offset in [-1, 0, 1]:
            offset = (x3_offset, x2_offset, 0)
            bid = get_buffer_id(*offset)
            if send_bufs[bid] is not None:
                part = block.part(offset)
                send_bufs[bid][:nhydro,:].copy_(block_vars["hydro_u"][part])

@torch.compile
def deserialize_2d(block: snapy.MeshBlock,
                   block_vars: dict[str, torch.Tensor],
                   recv_bufs: List[Optional[torch.Tensor]]):
    nhydro = block_vars["hydro_u"].shape[0]

    for x3_offset in [-1, 0, 1]:
        for x2_offset in [-1, 0, 1]:
            offset = (x3_offset, x2_offset, 0)
            bid = get_buffer_id(*offset)
            if recv_bufs[bid] is not None:
                part = block.part(offset, exterior=True)
                block_vars["hydro_u"][part].copy_(recv_bufs[bid][:nhydro,:])

@torch.compile
def slab_exchange(block: snapy.MeshBlock,
                  block_vars: dict[str, torch.Tensor],
                  ranks: List[int],
                  send_bufs: List[Optional[torch.Tensor]],
                  recv_bufs: List[Optional[torch.Tensor]]):
    ops = []
    serialize_2d(block, block_vars, send_bufs)

    if not dist.is_initialized():
        for r in range(1, len(ranks)):
            if send_bufs[r] is not None and ranks[r] == ranks[get_buffer_id(0, 0, 0)]:
                recv_bufs[r].copy_(send_bufs[r])
        deserialize_2d(block, block_vars, recv_bufs)
        return

    for r in range(1, len(ranks)):
        if send_bufs[r] is not None:
            ops.append(dist.P2POp(dist.isend, send_bufs[r], ranks[r]))
            ops.append(dist.P2POp(dist.irecv, recv_bufs[r], ranks[r]))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs: r.wait()

    deserialize_2d(block, block_vars, recv_bufs)

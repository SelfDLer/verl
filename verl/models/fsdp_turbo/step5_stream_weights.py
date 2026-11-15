"""Export CPU-offloaded Step5 expert shards without a whole-bank NPU all-gather."""

from functools import lru_cache

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


@lru_cache(maxsize=None)
def _cpu_group(ranks):
    return dist.new_group(ranks=list(ranks), backend="gloo", use_local_synchronization=True)


def _gather_cpu(shard, group):
    size = dist.get_world_size(group)
    result = torch.empty((size, *shard.shape), dtype=shard.dtype, device="cpu")
    # Use bytes so Gloo does not need a BF16 reduction or cast.
    dist.all_gather_into_tensor(result.view(torch.uint8).view(-1),
                                shard.view(torch.uint8).view(-1), group=group)
    return result


def _expert_weights(name, param, device):
    mesh = param.device_mesh
    dims = [p.dim if type(p) is Shard else None for p in param.placements]
    valid_placements = all(type(p) in (Shard, Replicate) for p in param.placements)
    if param.ndim != 3 or not valid_placements or sorted(d for d in dims if d is not None) != [0, 1]:
        raise ValueError(f"Unexpected Step5 expert layout for {name}: {param.shape}, {param.placements}")
    ep_axis, fsdp_axis = dims.index(0), dims.index(1)
    groups = []
    for axis in (ep_axis, fsdp_axis):
        ranks = dist.get_process_group_ranks(mesh.get_group(axis))
        if ranks != sorted(ranks):
            raise ValueError(f"Unsupported nonmonotone Step5 mesh for {name}: {ranks}")
        groups.append(_cpu_group(tuple(ranks)))
    ep_group, fsdp_group = groups
    local = param.to_local().detach()
    if (local.shape[0] * mesh.size(ep_axis) != param.shape[0]
            or local.shape[1] * mesh.size(fsdp_axis) != param.shape[1]
            or local.shape[2] != param.shape[2]):
        raise ValueError(f"Uneven Step5 expert shards for {name}: {local.shape}, {param.shape}")

    prefix, projection = name.rsplit(".", 1)
    for local_id in range(local.shape[0]):
        shard = local[local_id].to("cpu").contiguous()
        # EFSDP partitions rows of one projection. EP partitions expert IDs.
        complete = _gather_cpu(shard, fsdp_group).view(*param.shape[1:])
        bank = _gather_cpu(complete, ep_group)
        for ep_rank, weight in enumerate(bank):
            expert = ep_rank * local.shape[0] + local_id
            if projection == "gate_up_proj":
                gate, up = weight.chunk(2, dim=0)
                yield f"{prefix}.{expert}.gate_proj.weight", gate.to(device)
                yield f"{prefix}.{expert}.up_proj.weight", up.to(device)
            else:
                yield f"{prefix}.{expert}.down_proj.weight", weight.to(device)


def export_weights(params, device):
    for name, param in params.items():
        if name.endswith((".moe.experts.gate_up_proj", ".moe.experts.down_proj")):
            if not isinstance(param, DTensor):
                raise ValueError(f"Expected a distributed expert parameter for {name}")
            yield from _expert_weights(name, param, device)
        else:
            yield name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param

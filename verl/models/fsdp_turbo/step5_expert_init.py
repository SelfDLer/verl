"""Keep full Step5 expert tensors on CPU rank zero until the FSDP broadcast."""

import torch

from fsdp_turbo.models.step5.modeling_step5 import Step5Experts, Step5MoE


def prepare_expert_state(engine, module):
    rank = torch.distributed.get_rank()
    if rank == 0:
        print("[step5-init] preparing expert weights", flush=True)
    if rank != 0:
        for child in module.modules():
            if isinstance(child, Step5MoE):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    getattr(child, name).to_empty(device="meta")

    engine._model_adapter.prepare_fsdp(engine, module)
    if rank != 0:
        return {}

    # state_dict holds references to the populated CPU tensors. Only replace
    # the model's parameters, so EP and E-FSDP can compute shapes on meta.
    full_state = module.state_dict()
    for child in module.modules():
        if isinstance(child, Step5Experts):
            child.to_empty(device="meta")
    print("[step5-init] expert weights prepared", flush=True)
    return full_state


def allocate_rank0_expert_shards(module):
    if torch.distributed.get_rank() != 0:
        return
    # EP and E-FSDP have both partitioned the expert parameters by this point.
    # Dense parameters and buffers, including nonpersistent RoPE, retain values.
    device = torch.accelerator.current_accelerator()
    for child in module.modules():
        if isinstance(child, Step5Experts):
            child.to_empty(device=device)
    print("[step5-init] allocated local expert shards", flush=True)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP-Turbo hooks for Step5; keep other model engines independent of this dependency."""

import json
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch

from . import ModelAdapter

logger = logging.getLogger(__name__)


@torch.no_grad()
def restore_fp32_checkpoint_tensors(module, checkpoint_path):
    """Preserve checkpoint values for the model's FP32 parameters and buffers.

    The upstream dtype conversion casts all tensors to BF16 before restoring
    FP32 storage for norms, routing bias and indexer tensors. Widening that
    rounded value cannot recover the checkpoint's precision. Reload only these
    small tensors after conversion, leaving the large matmul weights in BF16.
    """
    from safetensors import safe_open

    root = Path(checkpoint_path)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard = defaultdict(list)
    for name, tensor in module.state_dict().items():
        if tensor.dtype == torch.float32 and name in weight_map:
            by_shard[weight_map[name]].append((name, tensor))
    for shard, tensors in by_shard.items():
        with safe_open(str(root / shard), framework="pt", device="cpu") as source:
            for name, tensor in tensors:
                value = source.get_tensor(name)
                if value.shape != tensor.shape:
                    raise ValueError(f"Checkpoint shape mismatch for {name}: {value.shape} != {tensor.shape}")
                tensor.copy_(value)


def configure_step5_vision_freezing(module, config):
    """Apply independent vision/projector freeze flags before FSDP and optimizer setup.

    Missing flags preserve existing trainability. False never unfreezes parameters
    frozen by another mechanism.
    """
    targets = (
        ("freeze_vision_encoder", "vision_model"),
        ("freeze_vision_projector", "vit_large_projector"),
    )
    for flag, name in targets:
        enabled = getattr(config, flag, False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{flag} must be a boolean, got {enabled!r}")
        if enabled and not isinstance(getattr(module, name, None), torch.nn.Module):
            raise ValueError(f"{flag} requires model submodule {name}")
    for flag, name in targets:
        if getattr(config, flag, False):
            target = getattr(module, name)
            target.requires_grad_(False)
            logger.info("Step5 froze %s (%s parameters)", name, sum(p.numel() for p in target.parameters()))


class Step5ModelAdapter(ModelAdapter):
    def build_module(self, engine, default_builder):
        if engine.engine_config.router_replay.mode not in ("disabled", "R3"):
            raise ValueError("Step5 FSDP-Turbo supports router_replay modes disabled and R3")
        distributed = engine.fsdp_turbo_config.distributed
        if engine.use_remove_padding and any(size != 1 for size in (
            distributed.tensor_parallel_size,
            distributed.ulysses_parallel_size,
            distributed.kcp_parallel_size,
            distributed.kv_allgather_parallel_size,
            engine.ulysses_sequence_parallel_size,
        )):
            raise ValueError("Step5 packed TND requires TP and context parallel sizes of 1")
        if distributed.expert_parallel_size > 1:
            config = engine.model_config.hf_config
            text_config = getattr(config, "text_config", config)
            if text_config.moe_num_experts % distributed.expert_parallel_size:
                raise ValueError("Step5 expert count must be divisible by expert_parallel_size")
            if distributed.ep_plan.dispatcher not in (
                "custom_native_eager_forward",
                "custom_native_eager_fused_forward",
                "custom_native_mc2_forward",
            ):
                raise ValueError("Step5 EP requires a Step5 native dispatcher to preserve FP32 combine")

        from fsdp_turbo.models.step5 import (
            Step5ForConditionalGeneration,
            configure_step5_checkpoint_expert_layout,
            convert_step5_model_dtype,
        )

        from verl.models.transformers.step5 import align_step5_vision_calls

        config = engine.model_config.hf_config
        configure_step5_checkpoint_expert_layout(config, engine.model_config.local_path)
        module = Step5ForConditionalGeneration.from_pretrained(
            engine.model_config.local_path, config=config, dtype=torch.bfloat16, trust_remote_code=False
        )
        convert_step5_model_dtype(module, torch.bfloat16)
        restore_fp32_checkpoint_tensors(module, engine.model_config.local_path)
        align_step5_vision_calls(module)
        configure_step5_vision_freezing(module, config)
        return module

    def prepare_lora(self, engine, module, config):
        """Infer expert adaptation from PEFT targets, without a second LoRA switch."""
        from fsdp_turbo.models.step5 import prepare_step5_model_for_fsdp
        from fsdp_turbo.models.step5.modeling_step5 import Step5MoE
        from peft.tuners.tuners_utils import check_target_module_exists

        expert_targets = any(
            check_target_module_exists(config, f"{name}.experts.{projection}")
            for name, child in module.named_modules()
            if isinstance(child, Step5MoE)
            for projection in ("gate_up_proj", "down_proj")
        )
        if expert_targets:
            distributed = engine.fsdp_turbo_config.distributed
            if distributed.expert_parallel_size > 1 and distributed.ep_plan.dispatcher not in (
                "custom_native_eager_forward",
                "custom_native_eager_fused_forward",
            ):
                raise ValueError("Step5 expert LoRA requires a native eager EP dispatcher")
            prepare_step5_model_for_fsdp(module, fuse_gate_up=True, grouped_projections=True)

    def prepare_fsdp(self, engine, module):
        from fsdp_turbo.models.step5 import configure_step5_fsdp_plan, prepare_step5_model_for_fsdp

        if hasattr(module, "peft_config"):
            if not engine.model_config.lora.get("merge", False):
                raise ValueError("Step5 LoRA requires model.lora.merge=true for rollout weight sync")
            allowed = (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "moe.experts.gate_up_proj",
                "moe.experts.down_proj",
            )
            adapters = [name for name, child in module.named_modules() if hasattr(child, "lora_A")]
            if not adapters or any(not name.endswith(allowed) for name in adapters):
                raise ValueError(
                    "Step5 LoRA supports only text self_attn q_proj/k_proj/v_proj/o_proj and routed experts"
                )

        distributed = engine.fsdp_turbo_config.distributed
        fuse_gate_up = distributed.expert_parallel_size > 1 and distributed.ep_plan.dispatcher in (
            "custom_native_eager_fused_forward",
            "custom_native_mc2_forward",
        )
        prepare_step5_model_for_fsdp(module, fuse_gate_up=fuse_gate_up)
        configure_step5_fsdp_plan(engine.fsdp_turbo_config.distributed.fsdp_plan)
        if hasattr(module, "peft_config"):
            # PEFT adds base_model.model to every base-model path. Keep the
            # caller's plan intact and preserve the PEFT root for disable_adapter
            # and merged weight export in the parent FSDP engine.
            config = deepcopy(engine.fsdp_turbo_config)
            prefix = "base_model.model."

            def prefixed(pattern):
                return pattern if pattern.startswith(prefix) else prefix + pattern

            fsdp = config.distributed.fsdp_plan
            fsdp.apply_modules = {prefixed(name): plan for name, plan in fsdp.apply_modules.items()}
            fsdp.ignored_modules = [prefixed(name) for name in (fsdp.ignored_modules or [])]
            fsdp.hook_modules = [prefixed(name) for name in (fsdp.hook_modules or [])]
            ep = config.distributed.ep_plan
            ep.apply_modules = [prefixed(name) for name in (ep.apply_modules or [])]
            ep.apply_efsdp_modules = [prefixed(name) for name in (ep.apply_efsdp_modules or [])]
            config.memory.recompute_plan = [prefixed(name) for name in (config.memory.recompute_plan or [])]
            for plan in (config.memory.chunk_batch_plan, config.memory.swap_activation_plan):
                if plan is not None:
                    plan.apply_modules = [prefixed(name) for name in (plan.apply_modules or [])]
            engine.fsdp_turbo_config = config

    def export_weights(self, weights):
        from verl.models.transformers.step5 import step5_hf_weight_name

        # The engine materializes each DTensor with full_tensor() before this
        # hook, gathering both the EP and E-FSDP dimensions. Restore the HF
        # stacked layout after that collective, never split a local shard.
        for name, tensor in weights:
            if name.endswith((".moe.experts.gate_up_proj.weight", ".moe.experts.down_proj.weight")):
                name = name.removesuffix(".weight")
            if name.endswith(".moe.experts.gate_up_proj"):
                if tensor.ndim != 3 or tensor.shape[1] % 2:
                    raise ValueError(f"Invalid Step5 fused expert shape: {name} {tuple(tensor.shape)}")
                gate, up = tensor.chunk(2, dim=1)
                base = name.removesuffix(".gate_up_proj")
                yield step5_hf_weight_name(f"{base}.gate_proj"), gate.contiguous()
                yield step5_hf_weight_name(f"{base}.up_proj"), up.contiguous()
            else:
                yield step5_hf_weight_name(name), tensor

    def prepare_model_inputs(self, engine, micro_batch, model_inputs):
        """Translate packed token boundaries and align optional rollout routes."""
        pad_size = 0
        if engine.use_remove_padding:
            tokens = micro_batch["input_ids"]
            packed_tokens = model_inputs["input_ids"]
            if not tokens.is_nested or packed_tokens.ndim != 2 or packed_tokens.shape[0] != 1:
                raise ValueError("Step5 remove padding requires nested tokens packed into [1, T]")
            offsets = tokens.offsets().to(device=packed_tokens.device, dtype=torch.long)
            valid_length = tokens.values().shape[0]
            pad_size = packed_tokens.shape[1] - valid_length
            if pad_size < 0:
                raise ValueError("Step5 packed inputs cannot be shorter than their token offsets")
            actual_seq_len = offsets[1:]
            if pad_size:
                # Isolate engine-added alignment padding as a separate document.
                actual_seq_len = torch.cat(
                    [actual_seq_len, actual_seq_len.new_tensor([packed_tokens.shape[1]])]
                )
                positions = model_inputs.get("position_ids")
                if positions is not None:
                    positions = positions.clone()
                    positions[..., valid_length:] = torch.arange(
                        pad_size, device=positions.device, dtype=positions.dtype
                    )
                    model_inputs["position_ids"] = positions
            model_inputs.pop("cu_seqlens", None)
            model_inputs.pop("cu_seqlens_cpu", None)
            model_inputs["actual_seq_len"] = actual_seq_len
            model_inputs["sfa_layout"] = "TND"
        if engine.engine_config.router_replay.mode == "disabled":
            return model_inputs
        routes = micro_batch.get("routed_experts")
        if routes is None:
            raise ValueError("Step5 R3 requires routed_experts; enable rollout.enable_rollout_routing_replay")
        if engine.use_remove_padding and not routes.is_nested:
            raise ValueError("Step5 packed R3 requires jagged routes with the same offsets as input_ids")
        if routes.is_nested:
            tokens = micro_batch["input_ids"]
            if not tokens.is_nested or not torch.equal(routes.offsets(), tokens.offsets()):
                raise ValueError("Step5 R3 routes and input_ids must have identical per-sample lengths")
            if engine.use_remove_padding:
                # Routes describe the current input token, before the label shift.
                routes = routes.values()
                if pad_size:
                    suffix = routes.new_full((pad_size, *routes.shape[1:]), -1)
                    routes = torch.cat((routes, suffix))
                routes = routes.unsqueeze(0)
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                routes = torch.nested.to_padded_tensor(
                    routes, padding=0, output_size=(batch_size, sequence_length, *routes.shape[2:])
                )
        model_inputs["routed_experts"] = routes
        response_mask = micro_batch.get("response_mask")
        sample_mask = None
        if response_mask is not None:
            # Synthetic padding samples have tokens but no policy loss or routes.
            sample_mask = (
                torch.stack([sample.bool().any() for sample in response_mask.unbind()])
                if response_mask.is_nested else response_mask.bool().any(dim=-1)
            )
            if sample_mask.shape != (micro_batch["input_ids"].size(0),):
                raise ValueError("Step5 routing replay requires one response mask per trajectory")
        if pad_size:
            if sample_mask is None:
                sample_mask = routes.new_ones(micro_batch["input_ids"].size(0), dtype=torch.bool)
            sample_mask = torch.cat((sample_mask, sample_mask.new_zeros(1)))
        if sample_mask is not None:
            model_inputs["routing_replay_sample_mask"] = sample_mask
        return model_inputs

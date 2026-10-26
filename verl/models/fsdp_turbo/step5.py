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
from collections import defaultdict
from pathlib import Path

import torch

from . import ModelAdapter


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


class Step5ModelAdapter(ModelAdapter):
    def build_module(self, engine, default_builder):
        if engine.use_remove_padding or engine._is_ulysses_enabled():
            raise ValueError("Step5 requires use_remove_padding=False and Ulysses size 1")
        if engine.fsdp_turbo_config.distributed.expert_parallel_size != 1:
            raise ValueError("Step5 RL weight synchronization currently requires expert_parallel_size=1")

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
        return module

    def prepare_fsdp(self, engine, module):
        from fsdp_turbo.models.step5 import configure_step5_fsdp_plan, prepare_step5_model_for_fsdp

        prepare_step5_model_for_fsdp(module)
        configure_step5_fsdp_plan(engine.fsdp_turbo_config.distributed.fsdp_plan)

    def export_weights(self, weights):
        from verl.models.transformers.step5 import step5_hf_weight_name

        return ((step5_hf_weight_name(name), tensor) for name, tensor in weights)

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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

"""PEFT adapter for expert-major GroupedLinear weights.

Register the custom module on both creation and reload; PEFT does not serialize
custom Python classes in adapter_config.json. Only classic LoRA is supported.
"""

import math

import torch
from fsdp_turbo.ops.grouped_linear import GroupedLinear
from fsdp_turbo.ops.moe import grouped_matmul
from peft.tuners.lora.layer import LoraLayer
from torch import nn
from torch.distributed.tensor import DTensor


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class GroupedLinearLoRA(nn.Module, LoraLayer):
    """Independent per-expert, per-output-partition classic low-rank updates."""

    def __init__(self, base_layer, adapter_name, r=0, lora_alpha=1, **kwargs):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, **kwargs)

    def _get_in_out_features(self, module):
        return module.in_features, module.out_features

    def update_layer(self, adapter_name, r, lora_alpha=1, config=None, **kwargs):
        """Create A/B with an explicit expert axis, also for adapter reload."""
        if config is not None:
            for key in ("lora_dropout", "use_dora", "use_rslora", "lora_bias", "init_lora_weights"):
                kwargs[key] = getattr(config, key)
        if r <= 0 or any(kwargs.get(key, False) for key in ("lora_dropout", "use_dora", "use_rslora", "lora_bias")):
            raise ValueError("GroupedLinear supports positive-rank classic LoRA without dropout, bias or variants")
        init = kwargs.get("init_lora_weights", True)
        if init is not True:
            raise ValueError("GroupedLinear requires the standard zero-B LoRA initialization")
        base = self.get_base_layer()
        groups = len(base.output_splits)
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r
        self.use_dora[adapter_name] = False
        self.use_rslora[adapter_name] = False
        self.lora_bias[adapter_name] = False
        self.lora_dropout[adapter_name] = nn.Identity()
        a = torch.empty(
            base.num_experts, groups * r, base.in_features, device=base.weight.device, dtype=base.weight.dtype
        )
        b = torch.zeros(base.num_experts, base.out_features, r, device=base.weight.device, dtype=base.weight.dtype)
        for expert in a:
            nn.init.kaiming_uniform_(expert, a=math.sqrt(5))
        self.lora_A[adapter_name] = GroupedLinear(nn.Parameter(a))
        self.lora_B[adapter_name] = GroupedLinear(nn.Parameter(b), base.output_splits)
        self.set_adapter(self.active_adapters)

    def forward(self, inputs, counts):
        """Reuse the already-dispatched tokens; no additional EP communication."""
        if self.disable_adapters and self.merged:
            self.unmerge()
        result = self.base_layer(inputs, counts)
        if self.disable_adapters or self.merged:
            return result
        for adapter in self.active_adapters:
            if adapter not in self.lora_A:
                continue
            hidden = self.lora_A[adapter](inputs, counts)
            b = _local(self.lora_B[adapter].weight)
            updates = []
            offset = 0
            for index, size in enumerate(self.get_base_layer().output_splits):
                rank = self.r[adapter]
                part = hidden[:, index * rank : (index + 1) * rank].contiguous()
                weight = b[:, offset : offset + size].contiguous()
                if inputs.shape[0] == 0:
                    update = inputs.new_empty((0, size)) + (part.sum() + weight.sum()) * 0
                else:
                    update = grouped_matmul(part, counts, weight, use_eager=inputs.device.type == "cpu")
                updates.append(update)
                offset += size
            result = result + torch.cat(updates, dim=-1) * self.scaling[adapter]
        return result

    def get_delta_weight(self, adapter):
        """Compute the delta with DTensor placement semantics during export."""
        a, b = self.lora_A[adapter].weight, self.lora_B[adapter].weight
        result, offset = [], 0
        for index, size in enumerate(self.get_base_layer().output_splits):
            rank = self.r[adapter]
            result.append(b[:, offset : offset + size] @ a[:, index * rank : (index + 1) * rank])
            offset += size
        return torch.cat(result, dim=1) * self.scaling[adapter]

    @torch.no_grad()
    def merge(self, safe_merge=False, adapter_names=None):
        """Merge using distributed operations, preserving EP/E-FSDP placement."""
        weight = self.get_base_layer().weight
        for adapter in adapter_names or self.active_adapters:
            if adapter in self.merged_adapters or adapter not in self.lora_A:
                continue
            delta = self.get_delta_weight(adapter).to(weight.dtype)
            if isinstance(delta, DTensor):
                delta = delta.redistribute(placements=weight.placements)
            merged = weight + delta
            if safe_merge and not torch.isfinite(merged).all():
                raise ValueError("Non-finite grouped LoRA merge")
            weight.copy_(merged)
            self.merged_adapters.append(adapter)

    @torch.no_grad()
    def unmerge(self):
        """Undo a merge; verl's export path restores an exact base backup instead."""
        weight = self.get_base_layer().weight
        while self.merged_adapters:
            delta = self.get_delta_weight(self.merged_adapters.pop()).to(weight.dtype)
            if isinstance(delta, DTensor):
                delta = delta.redistribute(placements=weight.placements)
            weight.sub_(delta)


def register_grouped_lora(config):
    """Register the custom expert layer on a PEFT configuration."""
    config._register_custom_module({GroupedLinear: GroupedLinearLoRA})
    return config

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

"""Expert LoRA correctness, partition independence, empty groups and reload."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("fsdp_turbo")
pytest.importorskip("peft")

from fsdp_turbo.ops.grouped_linear import GroupedLinear
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from verl.utils.peft_grouped_linear import register_grouped_lora


class ExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = GroupedLinear(nn.Parameter(torch.randn(3, 12, 8)), (6, 6))

    def forward(self, inputs, counts):
        return self.projection(inputs, counts)


@pytest.mark.parametrize("counts", [[2, 0, 3], [0, 0, 0]])
def test_grouped_lora_forward_gradients_and_checkpoint(counts, tmp_path):
    torch.manual_seed(42)
    base = ExpertModel()
    pristine = deepcopy(base)
    model = get_peft_model(base, register_grouped_lora(LoraConfig(r=2, lora_alpha=4, target_modules=["projection"])))
    counts = torch.tensor(counts)
    inputs = torch.randn(int(counts.sum()), 8, requires_grad=True)
    torch.testing.assert_close(model(inputs, counts), pristine(inputs, counts))
    layer = model.base_model.model.projection
    with torch.no_grad():
        layer.lora_B.default.weight.normal_(std=0.1)
    weight = pristine.projection.weight + layer.get_delta_weight("default")
    reference = torch.cat([x @ w.T for x, w in zip(inputs.split(counts.tolist()), weight, strict=True)])
    actual = model(inputs, counts)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    assert layer.base_layer.weight.grad is None
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert inputs.grad is not None
    # The unrouted middle expert receives zero gradient in both independent projections.
    assert not layer.lora_A.default.weight.grad[1].count_nonzero()
    assert not layer.lora_B.default.weight.grad[1].count_nonzero()
    with model.disable_adapter():
        torch.testing.assert_close(model(inputs, counts), pristine(inputs, counts))
    model.save_pretrained(tmp_path, safe_serialization=True)
    config = register_grouped_lora(LoraConfig.from_pretrained(tmp_path))
    restored = PeftModel.from_pretrained(deepcopy(pristine), tmp_path, config=config, is_trainable=True)
    torch.testing.assert_close(restored(inputs, counts), actual)
    merged = restored.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(merged(inputs, counts), actual, atol=2e-6, rtol=2e-5)


def test_gate_up_have_independent_input_factors():
    base = ExpertModel()
    model = get_peft_model(base, register_grouped_lora(LoraConfig(r=2, target_modules=["projection"])))
    layer = model.base_model.model.projection
    with torch.no_grad():
        layer.lora_B.default.weight.fill_(1)
        layer.lora_A.default.weight[:, :2].zero_()
        layer.lora_A.default.weight[:, 2:].fill_(1)
    delta = layer.get_delta_weight("default")
    assert not delta[:, :6].count_nonzero()
    assert delta[:, 6:].count_nonzero()

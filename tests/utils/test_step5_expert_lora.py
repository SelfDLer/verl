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

"""Step5 target selection and merged expert layout with real PEFT adapters."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fsdp_turbo")
pytest.importorskip("peft")

from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.models.step5 import Step5Config, Step5ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

from verl.models.fsdp_turbo.step5 import Step5ModelAdapter
from verl.utils.peft_grouped_linear import register_grouped_lora


@pytest.mark.parametrize("scope", ["attention", "experts", "both"])
def test_target_modules_select_scope_and_merge(scope):
    torch.manual_seed(42)
    config = Step5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=4,
        layer_types=["full_attention"],
        partial_rotary_factors=[1.0],
        use_moe=True,
        moe_layer_list=[0],
        moe_num_experts=2,
        moe_top_k=1,
        moe_intermediate_size=8,
        share_expert_dim=8,
    )
    model = Step5ForCausalLM(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    baseline = model(input_ids=tokens).logits.detach()
    attention = r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    experts = r".*\.moe\.experts\.(gate_up_proj|down_proj)"
    targets = {"attention": attention, "experts": experts, "both": f"({attention}|{experts})"}[scope]
    lora = register_grouped_lora(LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules=targets))
    engine = SimpleNamespace(fsdp_turbo_config=FSDPTurboConfig(), model_config=SimpleNamespace(lora={"merge": True}))
    engine.fsdp_turbo_config.distributed.fsdp_plan.apply_modules = {"model.layers.{*}": {}}
    adapter = Step5ModelAdapter()
    adapter.prepare_lora(engine, model, lora)
    model = get_peft_model(model, lora)
    adapter.prepare_fsdp(engine, model)
    torch.testing.assert_close(model(input_ids=tokens).logits, baseline, atol=2e-6, rtol=2e-5)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in name for name in trainable)
    assert any(".self_attn." in name for name in trainable) == (scope != "experts")
    assert any(".moe.experts." in name for name in trainable) == (scope != "attention")
    with torch.no_grad():
        for name, param in trainable.items():
            if ".lora_B." in name:
                param.normal_(std=0.05)
    output = model(input_ids=tokens).logits
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    merged = deepcopy(model).merge_and_unload()
    torch.testing.assert_close(merged(input_ids=tokens).logits, output, atol=3e-6, rtol=3e-5)
    exported = dict(adapter.export_weights(merged.state_dict().items()))
    assert not any("lora_" in name or "base_layer" in name or ".experts." in name for name in exported)
    assert exported["model.layers.0.moe.gate_proj.weight"].shape == (2, 8, 16)

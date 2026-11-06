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

"""Validate Step5 PEFT paths before any distributed wrapping or weight loading."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fsdp_turbo")
pytest.importorskip("peft")

from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.models.step5 import Step5Config, Step5ForCausalLM
from fsdp_turbo.utils.str_match import module_name_match
from peft import LoraConfig, TaskType, get_peft_model

from verl.models.fsdp_turbo.step5 import Step5ModelAdapter


def _model(targets):
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
    return get_peft_model(
        Step5ForCausalLM(config),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=targets,
        ),
    )


def _engine(merge=True):
    config = FSDPTurboConfig()
    config.distributed.fsdp_plan.apply_modules = {"model.layers.{*}": {}, "lm_head": {}}
    config.distributed.fsdp_plan.hook_modules = ["model.layers.{*}"]
    config.distributed.ep_plan.apply_modules = ["model.layers.{*}.moe.experts"]
    config.distributed.ep_plan.apply_efsdp_modules = ["model.layers.{*}.moe.experts"]
    config.memory.recompute_plan = ["model.layers.{*}"]
    return SimpleNamespace(fsdp_turbo_config=config, model_config=SimpleNamespace(lora={"merge": merge}))


def test_peft_prefix_preserves_fsdp_ep_and_recompute_targets():
    model = _model(["q_proj", "k_proj", "v_proj", "o_proj"])
    engine = _engine()
    original = engine.fsdp_turbo_config
    Step5ModelAdapter().prepare_fsdp(engine, model)
    config = engine.fsdp_turbo_config
    assert config is not original
    assert "model.layers.{*}" in original.distributed.fsdp_plan.apply_modules
    assert "base_model.model.model.layers.{*}" in config.distributed.fsdp_plan.apply_modules
    names = dict(model.named_modules())
    patterns = [
        *config.distributed.fsdp_plan.hook_modules,
        *config.distributed.ep_plan.apply_modules,
        *config.distributed.ep_plan.apply_efsdp_modules,
        *config.memory.recompute_plan,
    ]
    for pattern in patterns:
        assert any(module_name_match(pattern, name) for name in names), pattern
    assert not any(parameter.requires_grad for parameter in model.get_base_model().model.layers[0].moe.parameters())
    assert callable(model.disable_adapter)


@pytest.mark.parametrize(
    "targets,merge,match",
    [
        (["q_proj", "v_proj"], False, "merge=true"),
        (["gate"], True, "only text"),
        ("all-linear", True, "only text"),
    ],
)
def test_unsupported_lora_scope_fails_before_sharding(targets, merge, match):
    with pytest.raises(ValueError, match=match):
        Step5ModelAdapter().prepare_fsdp(_engine(merge), _model(targets))


def test_merged_export_keeps_expert_layout_conversion():
    tensor = torch.arange(48).reshape(2, 6, 4)
    weights = dict(
        Step5ModelAdapter().export_weights(
            [
                ("model.layers.0.moe.experts.gate_up_proj", tensor),
                ("model.layers.0.self_attn.q_proj.weight", torch.ones(16, 16)),
            ]
        )
    )
    assert "model.layers.0.self_attn.q_proj.weight" in weights
    assert len(weights) == 3
    assert all("experts" not in name for name in weights)

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

"""Storage precision at the Step5 RL checkpoint-to-optimizer boundary."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fsdp_turbo")
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.models.step5 import Step5ForConditionalGeneration, Step5VLConfig

from verl.models.fsdp_turbo.step5 import Step5ModelAdapter
from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer


@pytest.fixture(scope="module", params=["single", "sharded"])
def checkpoint(tmp_path_factory, request):
    path = tmp_path_factory.mktemp(request.param)
    config = Step5VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 4,
            "layer_types": ["sliding_attention"],
            "partial_rotary_factors": [1.0],
            "rope_theta": [10000.0],
            "sliding_window": 4,
            "num_nextn_predict_layers": 0,
            "use_moe": True,
            "moe_layer_list": [0],
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_intermediate_size": 8,
            "share_expert_dim": 8,
        },
        vision_config={"width": 8, "layers": 1, "heads": 2, "image_size": 8, "patch_size": 2},
        image_token_id=31,
    )
    model = Step5ForConditionalGeneration(config).float()
    with torch.no_grad():
        layer = model.model.layers[0]
        for tensor in (layer.self_attn.q_proj.weight, layer.moe.down_proj.weight):
            tensor.fill_(0.02001234)  # Not exactly representable in BF16.
            if request.param == "sharded":
                tensor.data = tensor.data.bfloat16()
        for tensor in (layer.input_layernorm.weight, layer.moe.router_bias):
            tensor.fill_(0.02001234)
    model.save_pretrained(path, max_shard_size="4KB" if request.param == "sharded" else "1GB")
    return path, layer.self_attn.q_proj.weight.flatten()[0].item()


@pytest.mark.parametrize(
    ("model_dtype", "forward_only", "expected"),
    [
        ("bf16", False, torch.bfloat16),
        ("fp32", False, torch.float32),
        (None, False, torch.float32),
        (None, True, torch.bfloat16),
    ],
)
def test_step5_master_dtype(checkpoint, model_dtype, forward_only, expected):
    path, matrix_value = checkpoint
    config = Step5VLConfig.from_pretrained(path)
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(
            model_dtype=model_dtype, forward_only=forward_only, router_replay=SimpleNamespace(mode="R3")
        ),
        model_config=SimpleNamespace(hf_config=config, local_path=str(path)),
        fsdp_turbo_config=FSDPTurboConfig(),
        use_remove_padding=False,
    )
    engine.fsdp_turbo_config.distributed.fsdp_plan.apply_modules = {}
    adapter = Step5ModelAdapter()
    model = adapter.build_module(engine, default_builder=None)
    layer = model.model.layers[0]
    for tensor in (layer.input_layernorm.weight, layer.moe.router_bias):
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, torch.full_like(tensor, 0.02001234))

    adapter.prepare_fsdp(engine, model)
    weights = [layer.self_attn.q_proj.weight, layer.moe.experts.down_proj]
    for weight in weights:
        assert weight.dtype == expected
        assert torch.equal(weight, torch.full_like(weight, matrix_value))

    if not forward_only:
        # Select the CPU implementation without probing accelerator foreach kernels.
        optim_config = FSDPOptimizerConfig(lr=1e-6, override_optimizer_config={"foreach": False})
        optimizer = build_optimizer(model.parameters(), optim_config)
        before = [weight.detach().clone() for weight in weights]
        for weight in weights:
            weight.grad = torch.ones_like(weight)
        optimizer.step()
        for weight, original in zip(weights, before, strict=True):
            assert optimizer.state[weight]["exp_avg"].dtype == expected
            assert optimizer.state[weight]["exp_avg_sq"].dtype == expected
            assert torch.equal(weight, original) == (expected == torch.bfloat16)

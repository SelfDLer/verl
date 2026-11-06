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

"""Four-NPU EP/E-FSDP test of Step5 LoRA merged export and backward."""

import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("fsdp_turbo")
pytest.importorskip("peft")

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass
from fsdp_turbo.models.step5 import Step5Config, Step5ForCausalLM, convert_step5_model_dtype
from fsdp_turbo.utils.device import get_accelerator_name, set_accelerator_compatible
from peft import LoraConfig, TaskType, get_peft_model

from verl.models.fsdp_turbo.step5 import Step5ModelAdapter
from verl.workers.engine.fsdp.fsdp_turbo_impl import FSDPTurboEngineWithLMHead


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 4, reason="launch with torchrun --nproc_per_node=4")
def test_step5_expert_lora_ep_efsdp_export_and_backward():
    """Real HCCL EP2/E-FSDP2, nonzero LoRA merge, exact base restore and backward."""
    if get_accelerator_name() != "npu":
        pytest.skip("This EP/E-FSDP integration test requires four NPUs")
    set_accelerator_compatible()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("hccl")
    torch.manual_seed(42)
    config = Step5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=4,
        layer_types=["full_attention"] * 2,
        partial_rotary_factors=[1.0] * 2,
        use_moe=True,
        moe_layer_list=[0, 1],
        moe_num_experts=2,
        moe_top_k=1,
        moe_intermediate_size=8,
        share_expert_dim=8,
    )
    model = Step5ForCausalLM(config)
    convert_step5_model_dtype(model, torch.bfloat16)
    turbo_config = _dict_to_dataclass(
        FSDPTurboConfig,
        {
            "distributed": {
                "fully_shard_parallel_size": 4,
                "expert_parallel_size": 2,
                "expert_fully_shard_parallel_size": 2,
                "expert_data_parallel_size": 1,
                "ep_plan": {
                    "apply_modules": ["model.layers.{*}.moe.experts"],
                    "apply_efsdp_modules": ["model.layers.{*}.moe.experts"],
                    "dispatcher": "custom_native_eager_fused_forward",
                },
                "fsdp_plan": {
                    "apply_modules": {"model.layers.{*}": {}, "lm_head": {}, "model.embed_tokens": {}},
                    "param_dtype": "bf16",
                    "reduce_dtype": "fp32",
                    "output_dtype": "bf16",
                    "fsdp_implementation": "native",
                },
            },
        },
    )
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.fsdp_turbo_config = turbo_config
    engine.model_config = SimpleNamespace(lora={"merge": True})
    engine._is_offload_param = False
    adapter = Step5ModelAdapter()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=r".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|moe\.experts\.(gate_up_proj|down_proj))",
    )
    adapter.prepare_lora(engine, model, lora_config)
    from verl.utils.peft_grouped_linear import register_grouped_lora

    model = get_peft_model(model, register_grouped_lora(lora_config))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)
                if ".lora_B." in name:
                    parameter.normal_(std=0.02)
    reference = dict(adapter.export_weights(copy.deepcopy(model).merge_and_unload().state_dict().items()))
    adapter.prepare_fsdp(engine, model)
    engine.module = FSDPTurbo(engine.fsdp_turbo_config, model.to(torch.accelerator.current_accelerator())).model
    assert all(".lora_" in name for name, parameter in engine.module.named_parameters() if parameter.requires_grad)
    before = {name: parameter.to_local().detach().cpu().clone() for name, parameter in engine.module.named_parameters()}
    exported = dict(adapter.export_weights(engine._merged_lora_per_tensor_param()))
    assert not any("lora_" in name or "base_layer" in name for name in exported)
    for name, expected in reference.items():
        # The existing rollout exporter casts DTensors (including FP32 norms) to BF16.
        torch.testing.assert_close(exported[name].cpu(), expected.to(exported[name].dtype), atol=0, rtol=0, msg=name)
    for name, parameter in engine.module.named_parameters():
        torch.testing.assert_close(parameter.to_local().cpu(), before[name], atol=0, rtol=0, msg=name)
    tokens = torch.tensor([[1, 2, 3, 4]], device=torch.accelerator.current_accelerator())
    engine.module(input_ids=tokens).logits.float().square().mean().backward()
    assert all(parameter.grad is not None for parameter in engine.module.parameters() if parameter.requires_grad)
    assert any(".moe.experts." in n and p.requires_grad for n, p in engine.module.named_parameters())
    print(json.dumps({"rank": rank, "status": "PASS", "exported_tensors": len(exported)}, sort_keys=True), flush=True)
    dist.destroy_process_group()

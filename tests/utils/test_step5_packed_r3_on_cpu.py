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

"""CPU coverage for the Step5 packed R3 training adapter."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.models.fsdp_turbo.step5 import Step5ModelAdapter
from verl.utils import torch_functional
from verl.workers.engine.fsdp.fsdp_turbo_impl import FSDPTurboEngineWithLMHead


def jagged(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def engine_and_data(bucket):
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.use_remove_padding = True
    engine.use_ulysses_sp = False
    engine.pad_to_length = bool(bucket)
    engine.pad_to_length_bucket = bucket
    engine._model_adapter = Step5ModelAdapter()
    engine.engine_config = SimpleNamespace(router_replay=SimpleNamespace(mode="R3"))
    rows = [torch.tensor(row) for row in ([1, 4, 7, 2, 9, 5, 6], [4, 8, 3], [9, 10, 6, 3, 2])]
    routes = [torch.tensor(routes, dtype=torch.int16).view(-1, 1, 2) for routes in (
        [[1, 3], [2, 1], [0, 2], [1, 0], [3, 2], [2, 0], [-1, -1]],
        [[0, 0]] * 3,
        [[3, 0], [1, 0], [0, 2], [1, 3], [0, 0]],
    )]
    data = TensorDict({
        "input_ids": jagged(rows),
        "position_ids": jagged([torch.arange(len(row)) for row in rows]),
        "routed_experts": jagged(routes),
        "temperature": torch.ones(3),
        "response_mask": torch.tensor([[1, 1, 1], [0, 0, 0], [1, 1, 0]]),
    }, batch_size=[3])
    return engine, data


@pytest.mark.parametrize("bucket", [0, 32])
def test_packed_r3_matches_individual_response_logprobs_and_gradients(bucket, monkeypatch):
    # Use verl's CPU logprob kernel even when the test host has accelerator wheels.
    monkeypatch.setattr(torch_functional, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setattr(torch_functional, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    step5 = pytest.importorskip("fsdp_turbo.models.step5")
    engine, data = engine_and_data(bucket)
    torch.manual_seed(73)
    config = step5.Step5Config(
        hidden_size=16, intermediate_size=24, num_hidden_layers=1, vocab_size=32,
        num_attention_heads=4, num_attention_groups=2, head_dim=4, layer_types=["full_attention"],
        partial_rotary_factors=[1.0], rope_theta=[10000.0], num_nextn_predict_layers=0,
        use_moe=True, moe_layer_list=[0], moe_num_experts=4, moe_top_k=2, moe_intermediate_size=8,
        share_expert_dim=8, enable_dsa_aux_loss=False, use_gbsa_full_sequence=True,
        use_dsa_all_select_fast_path=True, use_dsa_vectorized_aux_loss=True,
        sparse_config={"enabled": True, "proxy_dim": 8, "sparse_indexer_rope_dim": 4,
                       "sparse_indexer_num_heads": 4, "num_provider_groups": 2, "region_block_size": 8, "topk": 1},
    )
    model = step5.Step5ForCausalLM(config)
    reference = deepcopy(model)
    inputs, output_args = engine.prepare_model_inputs(data)
    assert inputs["actual_seq_len"].tolist() == ([7, 10, 15, 32] if bucket else [7, 10, 15])
    assert inputs["routing_replay_sample_mask"].tolist() == ([True, False, True, False] if bucket else [True, False, True])
    torch.testing.assert_close(inputs["routed_experts"][0, :15], data["routed_experts"].values())
    if bucket:
        assert (inputs["routed_experts"][0, 15:] == -1).all()
    output = model(**inputs)
    assert output.aux_loss is None
    logprobs = engine.prepare_model_outputs(output, output_args, data, None)["log_probs"].unbind()
    expected_loss, actual_loss = 0, 0
    offset = 0
    for index, (row, routes) in enumerate(zip(data["input_ids"].unbind(), data["routed_experts"].unbind())):
        expected = reference(input_ids=row[None], actual_seq_len=torch.tensor([len(row)]),
                             routed_experts=routes[None], routing_replay_sample_mask=torch.tensor([index != 1]))
        torch.testing.assert_close(output.logits[:, offset:offset + len(row)], expected.logits, atol=2e-6, rtol=2e-5)
        # A response target is scored by the preceding logit, including its first token.
        count = int(data["response_mask"][index].sum())
        if count:
            start = len(row) - count
            selected = expected.logits[0, start - 1:-1].log_softmax(-1).gather(1, row[start:, None]).squeeze(1)
            actual = logprobs[index][start - 1:-1]
            torch.testing.assert_close(actual, selected, atol=2e-6, rtol=2e-5)
            actual_loss = actual_loss - actual.sum()
            expected_loss = expected_loss - selected.sum()
        offset += len(row)
    actual_loss.backward()
    expected_loss.backward()
    for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-5)


def test_packed_r3_checks_offsets_and_masks_alignment_without_response_mask():
    engine, data = engine_and_data(32)
    routes = data["routed_experts"].values()
    data["routed_experts"] = jagged([routes[:6], routes[6:10], routes[10:]])
    with pytest.raises(ValueError, match="identical per-sample lengths"):
        engine.prepare_model_inputs(data)
    data["routed_experts"] = jagged([routes[:7], routes[7:10], routes[10:]])
    data["response_mask"] = jagged([torch.tensor([1, 1]), torch.tensor([0]), torch.tensor([1])])
    inputs, _ = engine.prepare_model_inputs(data)
    assert inputs["routing_replay_sample_mask"].tolist() == [True, False, True, False]
    del data["response_mask"]
    inputs, _ = engine.prepare_model_inputs(data)
    assert inputs["routing_replay_sample_mask"].tolist() == [True, True, True, False]


def test_bsnd_r3_keeps_per_sample_routes():
    engine, data = engine_and_data(0)
    engine.use_remove_padding = False
    inputs = engine._model_adapter.prepare_model_inputs(engine, data, {"input_ids": torch.zeros(3, 7)})
    expected = torch.nested.to_padded_tensor(data["routed_experts"], padding=0)
    torch.testing.assert_close(inputs["routed_experts"], expected)
    assert inputs["routing_replay_sample_mask"].tolist() == [True, False, True]
    assert "actual_seq_len" not in inputs

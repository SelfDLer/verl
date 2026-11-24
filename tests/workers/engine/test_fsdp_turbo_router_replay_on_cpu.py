# Copyright 2026 Huawei Technologies Co., Ltd.
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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.fsdp_turbo_impl import FSDPTurboEngineWithLMHead


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def batch_and_engine():
    ids = [torch.arange(5), torch.arange(3), torch.arange(2)]
    route_rows = [torch.arange(row.numel() * 6, dtype=torch.int16).view(-1, 3, 2) for row in ids]
    routes = nested(route_rows)
    data = TensorDict(
        {
            "input_ids": nested(ids),
            "position_ids": nested(ids),
            "routed_experts": routes,
            "response_mask": nested([torch.ones(2), torch.ones(1), torch.zeros(1)]),
        },
        batch_size=[3],
    )
    tu.assign_non_tensor(data, use_remove_padding=False, enable_routing_replay=True, temperature=1.0)
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.pad_to_length = False
    engine.mode = "eval"
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(router_replay={"mode": "R3"})
    return data, engine


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_replay_aligns_to_input_positions_and_masks_unrecorded_tokens(mode):
    data, engine = batch_and_engine()
    engine.mode = mode
    inputs, _ = engine.prepare_model_inputs(data)
    padded = inputs["routed_experts"]
    assert padded.shape == (3, 5, 3, 2)
    assert data["routed_experts"].dtype == torch.int16
    assert padded.dtype == torch.int32
    for row, valid in enumerate([4, 2, 0]):
        torch.testing.assert_close(padded[row, :valid], data["routed_experts"][row][:valid].int())
        assert (padded[row, valid:] == -1).all()


def test_reference_or_disabled_does_not_require_routes():
    data, engine = batch_and_engine()
    del data["routed_experts"]
    tu.assign_non_tensor(data, enable_routing_replay=False)
    inputs, _ = engine.prepare_model_inputs(data)
    assert "routed_experts" not in inputs


def test_missing_and_misaligned_routes_fail():
    data, engine = batch_and_engine()
    routes = data.pop("routed_experts")
    with pytest.raises(ValueError, match="requires routed_experts from rollout"):
        engine.prepare_model_inputs(data)
    data["routed_experts"] = nested([row[:-1] for row in routes.unbind()])
    with pytest.raises(ValueError, match="absolute input token"):
        engine.prepare_model_inputs(data)


def test_r3_rejects_model_that_accepts_but_ignores_route_kwargs():
    class KwargsMoE(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return input_ids

    _, engine = batch_and_engine()
    with pytest.raises(ValueError, match="KwargsMoE does not support.*R3"):
        engine._build_fsdp_module(KwargsMoE())

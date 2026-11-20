# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from torch import nn


@pytest.fixture
def bucket_worker(monkeypatch):
    pytest.importorskip("vllm.model_executor.model_loader.mtp_validation")
    from verl.workers.rollout.vllm_rollout import utils

    worker = object.__new__(utils.vLLMColocateWorkerExtension)
    worker.device = torch.device("cpu")
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16), speculative_config=None)
    worker.model_runner = SimpleNamespace(model=nn.Module(), vllm_config=config)
    monkeypatch.setattr(worker, "_get_zmq_handle", lambda: "cpu-test")
    monkeypatch.setattr(worker, "_iter_all_models_with_config", lambda: iter(()))
    receiver = Mock()
    monkeypatch.setattr(utils, "BucketedWeightReceiver", lambda **kwargs: receiver)
    monkeypatch.setattr(utils, "is_quantized_model", lambda _: False)
    return worker, receiver


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_step5_bucket_load_uses_scoped_check(bucket_worker, monkeypatch, dtype):
    step5 = pytest.importorskip("vllm.models.step5.nvidia.model")
    from vllm.model_executor.model_loader.mtp_validation import is_mtp_completeness_check_enabled

    class TinyProjection(step5.Step5FusedQKVIndexerLinear):
        def __init__(self):
            nn.Module.__init__(self)
            self.loaded_shard_ids = {}
            self.weight = nn.Parameter(torch.zeros(len(self.shard_ids), dtype=dtype), requires_grad=False)
            self.weight.weight_loader = self.load_shard

        def load_shard(self, param, tensor, shard_id):
            self.validate_shard_id(shard_id, param)
            param[self.shard_ids.index(shard_id)].copy_(tensor)

    # Use the real Step5 loaders and shard validator with one tiny fused projection.
    model = step5.Step5ForCausalLM.__new__(step5.Step5ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False, num_hidden_layers=1, num_attention_groups=2)
    model.model = step5.Step5Model.__new__(step5.Step5Model)
    nn.Module.__init__(model.model)
    model.model.config = model.config
    projection = TinyProjection()
    layer = nn.Module()
    layer.qkv_indexer_proj = projection
    model.model.layers = nn.ModuleList([layer])
    worker, receiver = bucket_worker
    worker.model_runner.model = model
    worker.model_runner.vllm_config.model_config.dtype = dtype
    load = Mock(wraps=model.load_weights)
    monkeypatch.setattr(model, "load_weights", load)
    names = ("q_proj", "k_proj", "v_proj", "sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_z", "g_proj")
    weights = [(f"model.layers.0.{name}.weight", torch.tensor(i + 1.0)) for i, name in enumerate(names)]

    def receive(on_bucket_received):
        on_bucket_received(weights[:3], False)
        on_bucket_received(weights[3:], True)

    receiver.receive_weights.side_effect = receive
    with pytest.raises(ValueError, match=r"Incomplete fused qkv\+indexer weight.*index_g"):
        worker.update_weights_from_ipc()
    assert load.call_count == 1

    load.reset_mock()
    projection.weight.zero_()
    worker.update_weights_from_ipc(skip_bucket_completeness_check=True)
    assert load.call_count == 2
    torch.testing.assert_close(projection.weight, torch.arange(1, 8, dtype=dtype))
    assert is_mtp_completeness_check_enabled()

    # An unrelated loader failure still propagates and restores the caller's context.
    load.side_effect = RuntimeError("bad weight")
    with pytest.raises(RuntimeError, match="bad weight"):
        worker.update_weights_from_ipc(skip_bucket_completeness_check=True)
    assert is_mtp_completeness_check_enabled()


@pytest.mark.parametrize(
    "peft,base_done,quantized,mtp,opt_in,check_enabled",
    [
        (None, False, False, False, False, True),
        (None, False, False, False, "false", True),
        ({"r": 4}, False, False, False, True, False),
        ({"r": 4}, True, False, False, True, True),
        (None, False, True, False, True, False),
        (None, False, True, True, False, True),
        (None, False, False, True, False, False),
    ],
)
def test_specialized_bucket_check_policy(
    bucket_worker, monkeypatch, peft, base_done, quantized, mtp, opt_in, check_enabled
):
    from vllm.model_executor.model_loader.mtp_validation import is_mtp_completeness_check_enabled

    from verl.workers.rollout.vllm_rollout import utils

    worker, receiver = bucket_worker
    monkeypatch.setattr(worker, "_iter_all_models", lambda: iter(()))
    monkeypatch.setattr(worker, "_use_mtp_drafter_weight_sync", lambda: mtp)
    monkeypatch.setattr(worker, "remove_lora", Mock(), raising=False)
    monkeypatch.setattr(utils, "is_quantized_model", lambda _: quantized)
    observed = []
    receiver.receive_weights.side_effect = lambda **_: observed.append(is_mtp_completeness_check_enabled())
    worker.update_weights_from_ipc(peft_config=peft, base_sync_done=base_done, skip_bucket_completeness_check=opt_in)
    assert observed == [check_enabled]
    assert is_mtp_completeness_check_enabled()


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ([], False),
        (["rollout@actor_rollout_ref.rollout=step5_vllm"], True),
        (
            [
                "rollout@actor_rollout_ref.rollout=step5_vllm",
                "actor_rollout_ref.rollout.checkpoint_engine.skip_bucket_completeness_check=false",
            ],
            False,
        ),
    ],
)
def test_rollout_recipe_controls_bucket_rpc(monkeypatch, overrides, expected):
    pytest.importorskip("vllm")
    from hydra import compose, initialize_config_module

    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import CheckpointEngineConfig
    from verl.workers.rollout.vllm_rollout import vllm_rollout

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        rollout = compose(config_name="ppo_trainer", overrides=overrides).actor_rollout_ref.rollout
    if overrides:
        assert rollout.name == "vllm"
    config = omega_conf_to_dataclass(rollout.checkpoint_engine, CheckpointEngineConfig)
    adapter = SimpleNamespace(
        config=SimpleNamespace(checkpoint_engine=config),
        _execute_method=AsyncMock(return_value=None),
        use_shm=True,
        zmq_handle="cpu-test",
        _has_server=False,
        replica_rank=0,
        rollout_rank=0,
    )
    sender = SimpleNamespace(async_send_weights=AsyncMock())
    monkeypatch.setattr(vllm_rollout, "BucketedWeightSender", lambda **kwargs: sender)
    asyncio.run(vllm_rollout.ServerAdapter.update_weights(adapter, iter(()), peft_config=None))
    rpc_kwargs = {"use_shm": True, "peft_config": None}
    if expected:
        rpc_kwargs["skip_bucket_completeness_check"] = True
    adapter._execute_method.assert_awaited_once_with("update_weights_from_ipc", non_block=True, kwargs=rpc_kwargs)

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

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _vocab_parallel_embedding_stubs():
    """Stubs standing in for vLLM's ``VocabParallelEmbedding``, or {} if vLLM is real.

    ``weight_update_utils`` imports that class at module scope for tied-alias
    detection, so loading it on CPU needs a placeholder. This file only exercises
    the buffer helpers, which never touch it.
    """
    try:
        importlib.import_module("vllm.model_executor.layers.vocab_parallel_embedding")
    except ImportError:
        pass
    else:
        return {}

    names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.vocab_parallel_embedding",
    ]
    stubs = {name: sys.modules.get(name) or types.ModuleType(name) for name in names}
    stubs[names[-1]].VocabParallelEmbedding = type("VocabParallelEmbedding", (), {})
    return stubs


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    stubs = _vocab_parallel_embedding_stubs()
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_weight_update_utils = _load_weight_update_utils()
apply_buffer_updates = _weight_update_utils.apply_buffer_updates
split_buffer_updates = _weight_update_utils.split_buffer_updates


def _load_vllm_rollout_utils():
    """Load vllm_rollout/utils.py with heavyweight deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do not
    leak into other tests; the loaded module keeps working since it binds the
    names it needs at import time.
    """
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        pass

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack
    fake_vllm_utils.resolve_weight_name = lambda model, name, names: name

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.is_quantized_model = lambda config: False
    fake_vllm_quant.load_quanted_weights = lambda weights, runner, is_drafter=False: weights
    fake_vllm_quant.prepare_quanted_weights_for_loading = lambda model: None
    fake_vllm_quant.process_quanted_weights_after_loading = lambda model, state: None

    fake_vllm_unquant = types.ModuleType("verl.utils.vllm.vllm_unquant_utils")
    fake_vllm_unquant.stage_unquantized_moe_params = lambda model: []
    fake_vllm_unquant.fold_unquantized_moe_params = lambda layers: contextlib.nullcontext()

    fake_rocm_expert_map = types.ModuleType("verl.utils.vllm.rocm_vllm_moe_expert_map")
    fake_rocm_expert_map.restore_moe_expert_maps = lambda model: None

    fake_bucketed_transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    fake_bucketed_transfer.BucketedWeightReceiver = None

    # NOTE: deliberately do NOT stub verl.plugin.platform. It is lightweight and
    # imports fine on CPU. verl.utils.device binds `get_platform` at import time, so
    # a `lambda: None` stub gets baked into verl.utils.device during this window;
    # because only the keys in `fakes` below are restored, that fake would leak
    # process-wide and later tests would crash in get_device_name() with
    # "'NoneType' object has no attribute 'device_name'".

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_quant_utils": fake_vllm_quant,
        "verl.utils.vllm.vllm_unquant_utils": fake_vllm_unquant,
        "verl.utils.vllm.rocm_vllm_moe_expert_map": fake_rocm_expert_map,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": fake_bucketed_transfer,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyBlock()])


class _FakeVllmConfig:
    def __init__(self, speculative_config=None):
        self.speculative_config = speculative_config


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None):
        self.model = inner_model
        self.vllm_config = _FakeVllmConfig(speculative_config=speculative_config)


def test_split_buffer_updates_routes_registered_buffers():
    model = _ToyModel()
    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32)),
    ]

    param_updates, buffer_updates, named_buffers = split_buffer_updates(model, weights)

    assert [name for name, _ in param_updates] == ["model.layers.0.linear.weight"]
    assert [name for name, _ in buffer_updates] == ["model.layers.0.e_score_correction_bias"]
    assert "model.layers.0.e_score_correction_bias" in named_buffers


def test_apply_buffer_updates_copies_buffer_values():
    model = _ToyModel()
    updates = [("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 1)]

    loaded = apply_buffer_updates(model, updates)

    assert loaded == 1
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    )


def test_apply_buffer_updates_ignores_non_buffer_weights():
    model = _ToyModel()
    weights = [("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32))]

    loaded = apply_buffer_updates(model, weights)

    assert loaded == 0
    assert torch.count_nonzero(model.model.layers[0].e_score_correction_bias) == 0


def test_vllm_update_weights_loads_params_and_buffers():
    model = _ToyModel()
    loaded_param_names = []
    apply_named_buffers = []

    def _fake_load_weights(weights):
        loaded_param_names.extend(name for name, _ in weights)

    model.load_weights = _fake_load_weights

    original_apply_buffer_updates = _vllm_rollout_utils.apply_buffer_updates

    def _spy_apply_buffer_updates(inner_model, buffer_updates, named_buffers=None):
        apply_named_buffers.append(named_buffers)
        return original_apply_buffer_updates(inner_model, buffer_updates, named_buffers=named_buffers)

    _vllm_rollout_utils.apply_buffer_updates = _spy_apply_buffer_updates

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    try:
        worker._update_weights(weights, peft_config=None, base_sync_done=False)
    finally:
        _vllm_rollout_utils.apply_buffer_updates = original_apply_buffer_updates

    assert loaded_param_names == ["model.layers.0.linear.weight"]
    assert apply_named_buffers and apply_named_buffers[0] is not None
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    )


def test_vllm_update_weights_syncs_buffers_to_mtp_drafter():
    """When an MTP drafter is synced, its registered buffers must be updated too."""
    main_model = _ToyModel()
    drafter_model = _ToyModel()
    main_model.load_weights = lambda weights: None
    drafter_model.load_weights = lambda weights: None

    class _SpecConfig:
        method = "mtp"
        draft_model_config = object()

    class _Drafter:
        def __init__(self, m):
            self.model = m

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.drafter = _Drafter(drafter_model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)


def _make_streaming_worker(monkeypatch, events, with_mtp=False):
    """Exercise the real worker callback using CPU parameters and fake transport."""
    main = _ToyModel()
    main.model.layers.append(_ToyBlock())
    models = [("main", main)]
    spec_config = None
    if with_mtp:
        draft = _ToyModel()
        draft.model.layers.append(_ToyBlock())
        models.append(("draft", draft))
        spec_config = types.SimpleNamespace(method="mtp", draft_model_config=object())

    for label, model in models:

        def load(weights, label=label, model=model):
            params = dict(model.named_parameters())
            for name, tensor in weights:
                params[name].data.copy_(tensor)
                events.append(("load", label, name))

        model.load_weights = load

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "unused-cpu-test"
    worker.model_runner = _FakeModelRunner(main, speculative_config=spec_config)
    worker.model_runner.vllm_config.model_config = object()
    if with_mtp:
        worker.model_runner.drafter = types.SimpleNamespace(model=draft)

    postprocess = types.ModuleType("vllm.model_executor.model_loader.utils")

    def finish(model, config, device):
        assert device == torch.device("cpu")
        label = next(label for label, candidate in models if candidate is model)
        events.append(("finish", label))

    postprocess.process_weights_after_loading = finish
    monkeypatch.setitem(sys.modules, postprocess.__name__, postprocess)
    return worker, models, postprocess


@pytest.mark.parametrize("with_mtp", [False, True])
def test_ipc_loads_each_bucket_before_reuse_then_finalizes_once(monkeypatch, with_mtp):
    events = []
    worker, models, _ = _make_streaming_worker(monkeypatch, events, with_mtp)
    labels = [label for label, _ in models]
    expected_events = []

    class Receiver:
        def __init__(self, **kwargs):
            assert kwargs == {"zmq_handle": "unused-cpu-test", "device": torch.device("cpu"), "use_shm": False}

        def receive_weights(self, on_bucket_received):
            bucket = torch.empty(4, 4)
            for layer_index, value in enumerate((11.0, 22.0)):
                name = f"model.layers.{layer_index}.linear.weight"
                bucket.fill_(value)
                on_bucket_received([(name, bucket)], is_last=layer_index == 1)
                expected_events.extend(("load", label, name) for label in labels)
                # The per-bucket callback must finish loading before transport
                # overwrites its storage, without waiting for the final bucket.
                assert events == expected_events
                bucket.fill_(-999.0)
                for _, model in models:
                    torch.testing.assert_close(
                        model.model.layers[layer_index].linear.weight,
                        torch.full((4, 4), value),
                    )
            events.append(("receive_done",))
            expected_events.append(("receive_done",))

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", Receiver)
    worker.update_weights_from_ipc()

    expected_events.extend(("finish", label) for label in labels)
    assert events == expected_events


@pytest.mark.parametrize("failure_stage", ["transport", "load", "finalize"])
def test_ipc_failure_propagates_without_premature_finalization(monkeypatch, failure_stage):
    events = []
    worker, models, postprocess = _make_streaming_worker(monkeypatch, events)

    if failure_stage == "load":

        def fail_load(weights):
            raise RuntimeError("load failed")

        models[0][1].load_weights = fail_load
    if failure_stage == "finalize":

        def fail_finish(model, config, device):
            events.append(("finish_failed",))
            raise RuntimeError("finalize failed")

        postprocess.process_weights_after_loading = fail_finish

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received(
                [("model.layers.0.linear.weight", torch.ones(4, 4))],
                is_last=failure_stage != "transport",
            )
            if failure_stage == "transport":
                raise RuntimeError("transport failed")
            events.append(("receive_done",))

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", Receiver)
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        worker.update_weights_from_ipc()

    assert not any(event[0] == "finish" for event in events)
    if failure_stage == "finalize":
        assert events[-2:] == [("receive_done",), ("finish_failed",)]
    else:
        assert not any(event[0] == "receive_done" for event in events)


def test_ipc_lora_keeps_complete_adapter_without_base_postprocessing(monkeypatch):
    events = []
    worker, _, _ = _make_streaming_worker(monkeypatch, events)
    worker.remove_lora = lambda adapter_id: events.append(("remove_lora", adapter_id))
    received = []

    def update(weights, peft_config, base_sync_done):
        assert peft_config == {"r": 4} and base_sync_done
        received.extend((name, tensor.clone()) for name, tensor in weights)
        events.append(("load_lora",))

    worker._update_weights = update

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            bucket = torch.ones(4)
            on_bucket_received([("adapter.a", bucket)], is_last=False)
            assert not received
            bucket.fill_(2.0)
            on_bucket_received([("adapter.b", bucket)], is_last=True)
            bucket.fill_(-999.0)

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", Receiver)
    worker.update_weights_from_ipc(peft_config={"r": 4}, base_sync_done=True)

    assert [name for name, _ in received] == ["adapter.a", "adapter.b"]
    torch.testing.assert_close(received[0][1], torch.ones(4))
    torch.testing.assert_close(received[1][1], torch.full((4,), 2.0))
    assert [event[0] for event in events] == ["remove_lora", "load_lora"]

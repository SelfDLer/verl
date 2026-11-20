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

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from verl.models.transformers.step5 import Step5HFModelConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig


def test_step5_registers_before_each_hf_load(monkeypatch):
    calls = []
    step5 = ModuleType("fsdp_turbo.models.step5")
    step5.Step5VLConfig = object()
    step5.Step5Processor = object()
    step5.register_step5 = lambda: calls.append("register")

    def register_processor(config, processor, *, exist_ok):
        assert (config, processor, exist_ok) == (step5.Step5VLConfig, step5.Step5Processor, True)
        calls.append("processor")

    monkeypatch.setattr("transformers.AutoProcessor.register", register_processor)
    monkeypatch.setitem(sys.modules, step5.__name__, step5)
    monkeypatch.setattr(HFModelConfig, "__post_init__", lambda self: calls.append("load"))
    config = OmegaConf.create({"_target_": "verl.models.transformers.step5.Step5HFModelConfig", "path": "model"})

    for _ in range(2):
        assert type(omega_conf_to_dataclass(config)) is Step5HFModelConfig
    assert calls == ["register", "processor", "load"] * 2


def test_step5_config_loads_image_processor(tmp_path):
    pytest.importorskip("fsdp_turbo", exc_type=ModuleNotFoundError)
    from fsdp_turbo.models.step5 import Step5Processor, Step5VLConfig
    from fsdp_turbo.models.step5.fsdp import Step5FSDPTurboConfig
    from PIL import Image
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    markers = ["<im_patch>", "<im_start>", "<im_end>", "<patch_start>", "<patch_end>", "<patch_newline>"]
    vocab = {token: index for index, token in enumerate(["<unk>", "image", *markers])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.add_special_tokens(markers)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<unk>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(tmp_path)
    Step5VLConfig(
        architectures=["Step5ForConditionalGeneration"],
        vision_config={"image_size": 28},
        image_token_id=vocab["<im_patch>"],
        image_token_len=3,
        patch_token_len=2,
    ).save_pretrained(tmp_path)

    config = OmegaConf.create({"_target_": "verl.models.transformers.step5.Step5HFModelConfig", "path": str(tmp_path)})
    model = omega_conf_to_dataclass(config)
    assert model.fsdp_turbo_config_class is Step5FSDPTurboConfig
    assert isinstance(model.processor, Step5Processor)
    text = model.processor.apply_chat_template([{"role": "user", "content": "<im_patch>image"}], tokenize=False)
    inputs = model.processor(text=text, images=Image.new("RGB", (28, 28)), return_tensors="pt")
    assert inputs.pixel_values.shape == (1, 3, 28, 28)
    prompt_ids = model.processor.prepare_rollout_prompt_ids(inputs.input_ids[0].tolist())
    assert prompt_ids == [vocab["<im_patch>"], vocab["image"]]


@pytest.fixture
def turbo_builder(monkeypatch):
    build = Mock()
    for module_name, symbols in (
        ("fsdp_turbo.models.step5.fsdp", {"build_step5_fsdp_model": build}),
        ("fsdp_turbo.models.utils", {"get_model_by_name": lambda name: None}),
        ("fsdp_turbo.utils.model", {"set_fused_overrides": lambda config, turbo: None}),
    ):
        module = ModuleType(module_name)
        vars(module).update(symbols)
        monkeypatch.setitem(sys.modules, module_name, module)
    return build


@pytest.mark.parametrize(
    ("rank", "model_dtype", "forward_only", "expected_dtype", "device"),
    [
        (0, "bf16", False, torch.bfloat16, "cpu"),
        (1, None, True, torch.bfloat16, "meta"),
        (0, None, False, torch.float32, "cpu"),
    ],
)
def test_step5_fsdp_turbo_build(monkeypatch, turbo_builder, rank, model_dtype, forward_only, expected_dtype, device):
    from verl.workers.engine.fsdp import FSDPTurboEngineWithLMHead

    monkeypatch.setattr(Step5HFModelConfig, "__post_init__", lambda self: None)
    config = Step5HFModelConfig(
        path="checkpoint",
        local_path="checkpoint",
        hf_config=object(),
        use_remove_padding=False,
        enable_gradient_checkpointing=not forward_only,
    )
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.rank = rank
    engine.model_config = config
    engine.engine_config = SimpleNamespace(
        model_dtype=model_dtype, forward_only=forward_only, ulysses_sequence_parallel_size=1
    )
    distributed = SimpleNamespace(ulysses_parallel_size=1)
    engine.fsdp_turbo_config = SimpleNamespace(distributed=distributed)
    model = engine._build_module()
    turbo_builder.assert_called_once_with(
        "checkpoint", config.hf_config, distributed, dtype=expected_dtype, init_device=device
    )
    assert model is turbo_builder.return_value
    if forward_only:
        model.gradient_checkpointing_enable.assert_not_called()
    else:
        model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def test_fsdp_turbo_keeps_default_loader(monkeypatch, turbo_builder):
    from verl.workers.engine.fsdp import FSDPEngineWithLMHead, FSDPTurboEngineWithLMHead

    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.model_config = SimpleNamespace(hf_config=object())
    engine.fsdp_turbo_config = SimpleNamespace(model=SimpleNamespace(model_name_or_path="checkpoint"))
    engine.ulysses_sequence_parallel_size = 2
    loaded = object()
    monkeypatch.setattr(FSDPEngineWithLMHead, "_build_module", lambda self: loaded)
    assert engine._build_module() is loaded
    assert engine.ulysses_sequence_parallel_size == 2
    turbo_builder.assert_not_called()


@pytest.mark.parametrize(
    "use_step5, peft_config, base_sync_done, convert",
    [
        (True, None, True, True),
        (True, {"r": 4}, False, True),
        (True, {"r": 4}, True, False),
        (False, None, False, False),
    ],
)
def test_fsdp_turbo_hf_export(monkeypatch, use_step5, peft_config, base_sync_done, convert):
    from verl.workers.engine.fsdp import FSDPEngineWithLMHead, FSDPTurboEngineWithLMHead

    key = "model.layers.1.moe.experts.up_proj"
    tensor = torch.zeros(2, 1, 4)
    weights = iter([(key, tensor)])
    parent = Mock(return_value=(weights, peft_config))
    monkeypatch.setattr(FSDPEngineWithLMHead, "get_per_tensor_param", parent)
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.model_config = Step5HFModelConfig if use_step5 else SimpleNamespace()
    if convert:
        pytest.importorskip("fsdp_turbo.models.step5.checkpoint_compat")
    result, config = engine.get_per_tensor_param(layered_summon=True, base_sync_done=base_sync_done)
    assert config is peft_config
    parent.assert_called_once_with(True, base_sync_done)
    name, actual = next(result)
    assert name == ("model.layers.1.moe.up_proj.weight" if convert else key)
    assert actual is tensor

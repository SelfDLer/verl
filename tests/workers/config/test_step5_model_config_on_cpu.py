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
from types import ModuleType

import pytest
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
    assert isinstance(model.processor, Step5Processor)
    text = model.processor.apply_chat_template([{"role": "user", "content": "<im_patch>image"}], tokenize=False)
    inputs = model.processor(text=text, images=Image.new("RGB", (28, 28)), return_tensors="pt")
    assert inputs.pixel_values.shape == (1, 3, 28, 28)
    prompt_ids = model.processor.prepare_rollout_prompt_ids(inputs.input_ids[0].tolist())
    assert prompt_ids == [vocab["<im_patch>"], vocab["image"]]

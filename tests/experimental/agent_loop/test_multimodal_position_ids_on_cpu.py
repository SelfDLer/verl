# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import torch
from transformers import AutoConfig, AutoProcessor

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils.tokenizer import hf_processor


def test_compute_position_ids_accepts_processor_rope_kwargs_hook():
    class Processor:
        def get_rope_index_kwargs(self, multi_modal_inputs):
            return {"audio_seqlens": multi_modal_inputs["feature_attention_mask"].sum(-1)}

        def get_rope_index(
            self,
            *,
            input_ids,
            attention_mask,
            image_grid_thw,
            video_grid_thw,
            audio_seqlens,
        ):
            del attention_mask, image_grid_thw, video_grid_thw
            torch.testing.assert_close(audio_seqlens, torch.tensor([2]))
            return torch.zeros((3, 1, input_ids.shape[-1]), dtype=torch.long), None

    worker = type("Worker", (), {"processor": Processor()})()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {"feature_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long)}

    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids,
        attention_mask,
        multi_modal_inputs,
    )

    assert position_ids.shape == (1, 4, 3)
    torch.testing.assert_close(position_ids[:, 0], torch.tensor([[0, 1, 2]]))


def test_step5_processor_uses_standard_positions(monkeypatch):
    class Step5Processor:
        pass

    processor = Step5Processor()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    assert hf_processor("test-model") is processor
    ids = torch.tensor([[0, 0, 31, 11, 12]])
    positions = AgentLoopWorker._compute_position_ids(SimpleNamespace(processor=processor), ids, ids.ne(0).long(), {})
    torch.testing.assert_close(positions, torch.tensor([[0, 0, 0, 1, 2]]))


def test_step5_actor_requests_only_visual_inputs():
    from transformers.feature_extraction_utils import BatchFeature

    images = [object()]
    pixels = torch.ones(1, 3, 2, 2)

    class Step5Processor:
        def __call__(self, *, text, images, enable_patch, **kwargs):
            assert text is None and images is output.multi_modal_data["images"]
            assert enable_patch is False
            return BatchFeature({"pixel_values": pixels})

    output = SimpleNamespace(multi_modal_data={"images": images}, mm_processor_kwargs={"enable_patch": False})
    worker = SimpleNamespace(processor=Step5Processor())
    features = AgentLoopWorker._compute_multi_modal_inputs(worker, output, torch.tensor([[1, 2]]))
    torch.testing.assert_close(features["pixel_values"], pixels)

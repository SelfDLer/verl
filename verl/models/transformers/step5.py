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

"""Step5 multimodal bridge for verl; uses the FSDPTurbo model implementation."""

import copy
import types

from fsdp_turbo.models.step5 import Step5Processor, Step5VLConfig, register_step5
from transformers import AutoConfig, AutoProcessor

register_step5()


class Step5RLProcessor(Step5Processor):
    """Render structured RL messages using the released image-token protocol."""

    uses_standard_rope = True

    def __init__(self, tokenizer, config):
        super().__init__(tokenizer, config)
        self.config = config
        self.chat_template = tokenizer.chat_template
        # verl detects multimodal processors using the HF attribute name.
        self.image_processor = self.image_preprocessor

    @classmethod
    def from_pretrained(cls, name_or_path, **kwargs):
        from verl.utils.tokenizer.tokenizer import hf_tokenizer

        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(name_or_path, **kwargs)
        kwargs.setdefault("fix_mistral_regex", False)
        return cls(hf_tokenizer(name_or_path, **kwargs), config)

    def get_multi_modal_features(self, *, images=None, videos=None, audio=None, **kwargs):
        """Return media tensors independently of rollout text tokenization."""
        if videos or audio:
            raise ValueError("Step5 RL supports images and text only")
        return self.get_image_features(images) if images else {}

    def apply_chat_template(self, conversation, *args, **kwargs):
        messages = copy.deepcopy(conversation)
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item["type"] == "text":
                        parts.append(item["text"])
                    elif item["type"] in ("image", "image_url"):
                        parts.append(self.image_token)
                    else:
                        raise ValueError(f"Step5 RL does not support modality {item['type']}")
                message["content"] = "".join(parts)
        return self.tokenizer.apply_chat_template(messages, *args, **kwargs)

    def __call__(self, *args, videos=None, audio=None, **kwargs):
        if videos is not None or audio is not None:
            raise ValueError("Step5 RL currently supports images and text only")
        return super().__call__(*args, **kwargs)

    def save_pretrained(self, directory, **kwargs):
        self.config.save_pretrained(directory)
        return self.tokenizer.save_pretrained(directory, **kwargs)

    def prepare_rollout_prompt_ids(self, prompt_ids):
        from .step5_image_tokens import collapse_image_tokens

        names = ('<im_patch>', '<im_start>', '<im_end>', '<patch_start>', '<patch_end>', '<patch_newline>')
        token_ids = {name: self.tokenizer.convert_tokens_to_ids(name) for name in names}
        return collapse_image_tokens(list(prompt_ids), token_ids, self.num_image_feature_size, self.num_patch_feature_size)

    def get_image_features(self, images):
        """Rebuild vision tensors without decoding the generated token stream."""
        features = super().__call__(text=[self.image_token * len(images)], images=images, return_tensors="pt")
        return {key: value for key, value in features.items() if key not in ("input_ids", "attention_mask")}


def align_step5_vision_calls(model):
    """Keep FSDP collectives aligned for batches with differing crop counts."""
    if getattr(model, "_verl_vision_aligned", False):
        return
    original = model.get_multimodal_embeddings

    def aligned(self, pixel_values=None, patch_pixel_values=None, num_patches=None, image_embeds=None):
        features = original(pixel_values, patch_pixel_values, num_patches, image_embeds)
        if (
            pixel_values is not None
            and image_embeds is None
            and (patch_pixel_values is None or patch_pixel_values.numel() == 0)
        ):
            dummy = self._encode_images(pixel_values.new_zeros((1, 3, 504, 504)))
            features = features + dummy.float().sum().to(features.dtype) * 0.0
        return features

    model.get_multimodal_embeddings = types.MethodType(aligned, model)
    model._verl_vision_aligned = True


def step5_hf_weight_name(name):
    """Restore the released stacked-expert names for vLLM weight loading."""
    for projection in ("gate_proj", "up_proj", "down_proj"):
        suffix = f".moe.experts.{projection}"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".moe.{projection}.weight"
    return name


AutoProcessor.register(Step5VLConfig, Step5RLProcessor, exist_ok=True)

# Copyright 2026 Individual Contributor
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

"""Useful model FLOPs for Step5 full-parameter actor training.

Counts multiply-add as two FLOPs, backward as twice forward for trainable
matmuls. Counts causal/SWA/DSA token pairs, activated experts, and real image
crops. Excludes recomputation, dummy vision calls, padding, optimizer,
normalization, activations, routing/top-k and communication (MFU, not HFU).
The current PPO loss does not consume Step5's aux_loss: DSA indexer projections,
region scores and detached teacher QK scores are forward-only. MTP metadata does not
represent executed layers in the current FSDPTurbo Step5 actor.
"""

import math
from collections import Counter


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def step5_image_shapes(batch_inputs):
    """Read only tensor shapes; never copy image contents or concatenate pixels."""
    shapes = []
    for inputs in batch_inputs:
        if hasattr(inputs, "data") and not isinstance(inputs, dict):
            inputs = inputs.data
        if not inputs or inputs.get("image_embeds") is not None:
            continue
        for key in ("pixel_values", "patch_pixel_values"):
            pixels = inputs.get(key)
            if pixels is None:
                continue
            if pixels.ndim < 4:
                raise ValueError(f"Step5 {key} must have at least four dimensions")
            count = math.prod(pixels.shape[:-3])
            shapes.extend([tuple(pixels.shape[-2:])] * count)
    return shapes


def _causal_pairs(length, window=None):
    width = length if window is None else min(length, window)
    return width * (width + 1) // 2 + (length - width) * width


def _region_pairs(length, block, topk=None):
    """Sum completed (optionally selected) regions over all query positions."""
    blocks, tail = divmod(length, block)
    if topk is None or blocks <= topk:
        return block * blocks * (blocks - 1) // 2 + tail * blocks
    return block * (topk * (topk - 1) // 2 + (blocks - topk) * topk) + tail * topk


def _vision_flops(config, text_hidden, image_shapes):
    width = config.width
    patch = config.patch_size
    mlp = int(width * config.mlp_ratio)
    total = 0
    for (height, image_width), count in Counter(map(tuple, image_shapes)).items():
        if height <= 0 or image_width <= 0 or height % patch or image_width % patch:
            raise ValueError("Step5 image dimensions must be positive multiples of patch_size")
        h, w = height // patch, image_width // patch
        tokens = h * w + int(_get(config, "use_cls_token", False))
        # Patch input has no gradient; convolution has forward + weight gradient.
        flops = 4 * h * w * config.num_channels * patch**2 * width
        flops += config.layers * (6 * tokens * (4 * width**2 + 2 * width * mlp) + 12 * tokens**2 * width)
        h, w = (h + 1) // 2, (w + 1) // 2
        flops += 6 * h * w * 9 * width * (2 * width)
        h, w = (h + 1) // 2, (w + 1) // 2
        flops += 6 * h * w * 9 * (2 * width) * (4 * width)
        flops += 6 * h * w * (4 * width) * text_hidden
        total += count * flops
    return total


def estimate_step5_flops(config, tokens_sum, batch_seqlens, delta_time, **kwargs):
    """Return estimated training TFLOP/s, before division by device count/peak."""
    # This estimator is for actor updates only; preserve the previous inference metric.
    if kwargs.get("forward_only", False):
        return 0.0
    if delta_time <= 0:
        raise ValueError("Step5 MFU requires positive elapsed time")
    text = _get(config, "text_config", config)
    hidden = text.hidden_size
    heads, kv_heads, dim = text.num_attention_heads, text.num_attention_groups, text.head_dim
    layers = text.num_hidden_layers
    layer_types = list(text.layer_types)[:layers]
    if len(layer_types) != layers or any(t not in ("full_attention", "sliding_attention") for t in layer_types):
        raise ValueError("Step5 MFU requires a supported attention type for every decoder layer")
    if any(length < 0 for length in batch_seqlens) or tokens_sum != sum(batch_seqlens):
        raise ValueError("Invalid Step5 token lengths")
    moe_layers = set(text.moe_layer_list) if text.use_moe else set()
    sparse = _get(text, "sparse_config")
    # Embedding lookup is not a dense matmul. Count the LM head once, even when tied.
    linear_params = hidden * text.vocab_size
    attention_flops = 0
    indexer_forward_params = 0
    lengths = Counter(batch_seqlens)
    for index, layer_type in enumerate(layer_types):
        linear_params += hidden * (2 * heads + 2 * kv_heads) * dim
        if _get(text, "use_head_wise_attn_gate", True):
            linear_params += hidden * heads
        if index in moe_layers:
            linear_params += hidden * text.moe_num_experts
            linear_params += 3 * hidden * (text.moe_top_k * text.moe_intermediate_size + text.share_expert_dim)
        else:
            linear_params += 3 * hidden * text.intermediate_size
        is_sparse = (
            sparse
            and _get(sparse, "enabled", True)
            and layer_type in _get(sparse, "apply_to_layer_types", ["full_attention"])
        )
        if is_sparse:
            proxy, index_heads = _get(sparse, "proxy_dim"), _get(sparse, "sparse_indexer_num_heads")
            block, topk = _get(sparse, "region_block_size"), _get(sparse, "topk")
            if block <= 0 or topk < 0:
                raise ValueError("Invalid Step5 sparse region size/topk")
            indexer_forward_params += hidden * ((index_heads + 2) * proxy + index_heads)
        for length, count in lengths.items():
            if is_sparse:
                regions = _region_pairs(length, block)
                local_pairs = _causal_pairs(length) - block * regions
                pairs = local_pairs + block * _region_pairs(length, block, topk)
                # PPO consumes logits, not aux_loss; top-k indices do not propagate gradients.
                attention_flops += count * (2 * index_heads * proxy * regions + 2 * heads * dim * block * regions)
            else:
                window = text.sliding_window if layer_type == "sliding_attention" else None
                if window is not None and window <= 0:
                    raise ValueError("Step5 sliding window must be positive")
                pairs = _causal_pairs(length, window)
            attention_flops += count * 12 * heads * dim * pairs
    total = (6 * linear_params + 2 * indexer_forward_params) * tokens_sum + attention_flops
    image_shapes = kwargs.get("step5_image_shapes") or []
    if image_shapes:
        total += _vision_flops(config.vision_config, hidden, image_shapes)
    return total / delta_time / 1e12

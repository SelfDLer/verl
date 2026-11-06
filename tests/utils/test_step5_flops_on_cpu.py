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

from types import SimpleNamespace

import pytest
import torch
from tensordict import NonTensorData

from verl.utils.flops_counter import FlopsCounter, get_device_flops
from verl.utils.step5_flops import _causal_pairs, _region_pairs, estimate_step5_flops, step5_image_shapes


def config(**overrides):
    values = dict(
        model_type="step5",
        hidden_size=8,
        vocab_size=16,
        num_attention_heads=2,
        num_attention_groups=1,
        head_dim=4,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        sliding_window=2,
        use_head_wise_attn_gate=False,
        use_moe=False,
        moe_layer_list=[],
        intermediate_size=12,
        moe_num_experts=8,
        moe_top_k=2,
        moe_intermediate_size=4,
        share_expert_dim=6,
        sparse_config=None,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 17, 4104, 4105, 6144])
def test_pair_counts_against_enumeration(length):
    assert _causal_pairs(length) == sum(q + 1 for q in range(length))
    assert _causal_pairs(length, 512) == sum(min(q + 1, 512) for q in range(length))
    for topk in (0, 1, 512):
        regions = _region_pairs(length, 8, topk)
        assert regions == sum(min(q // 8, topk) for q in range(length))
        pairs = _causal_pairs(length) - 8 * _region_pairs(length, 8) + 8 * regions
        assert pairs == sum(8 * min(q // 8, topk) + q % 8 + 1 for q in range(length))


def test_dense_and_sliding_by_hand():
    # 128 LM head + 192 QKVO + 288 SwiGLU matrix elements; 6 causal pairs.
    assert estimate_step5_flops(config(), 3, [3], 1) * 1e12 == pytest.approx(6 * 608 * 3 + 12 * 8 * 6)
    sliding = config(layer_types=["sliding_attention"])
    assert estimate_step5_flops(sliding, 3, [3], 1) * 1e12 == pytest.approx(6 * 608 * 3 + 12 * 8 * 5)


def test_moe_activated_experts_shared_router_and_mtp():
    cfg = config(use_moe=True, moe_layer_list=[0], layer_types=["full_attention", "sliding_attention"])
    # Count 2 selected experts, shared expert and router, not all 8 experts.
    params = 128 + 192 + 8 * 8 + 3 * 8 * (2 * 4 + 6)
    expected = 6 * params * 3 + 12 * 8 * 6
    assert estimate_step5_flops(cfg, 3, [3], 1) * 1e12 == pytest.approx(expected)
    cfg.moe_num_experts = 16
    assert estimate_step5_flops(cfg, 3, [3], 1) * 1e12 - expected == pytest.approx(6 * 8 * 8 * 3)


@pytest.mark.parametrize("as_dict", [False, True])
def test_sparse_attention_and_detached_teacher(as_dict):
    sparse = dict(
        enabled=True,
        proxy_dim=2,
        sparse_indexer_num_heads=2,
        region_block_size=2,
        topk=1,
        apply_to_layer_types=["full_attention"],
    )
    cfg = config(sparse_config=sparse if as_dict else SimpleNamespace(**sparse))
    # length=5: 4 visible completed regions, 3 selected regions, 7 local pairs.
    # 80 indexer projection elements; indexer and teacher are forward-only for PPO.
    expected = (6 * 608 + 2 * 80) * 5 + 12 * 8 * 13 + 2 * 2 * 2 * 4 + 2 * 8 * 2 * 4
    assert estimate_step5_flops(cfg, 5, [5], 1) * 1e12 == pytest.approx(expected)


def test_vision_convolutions_attention_and_projector():
    cfg = SimpleNamespace(
        model_type="step5_vl",
        text_config=config(),
        vision_config=SimpleNamespace(
            width=4,
            patch_size=2,
            mlp_ratio=2,
            num_channels=3,
            layers=1,
            use_cls_token=False,
        ),
    )
    # 8x8 image: 16 ViT tokens -> 4 -> 1 downsampled tokens.
    expected = 4 * 16 * 3 * 4 * 4 + 6 * 16 * (4 * 16 + 2 * 4 * 8) + 12 * 256 * 4
    expected += 6 * 4 * 9 * 4 * 8 + 6 * 9 * 8 * 16 + 6 * 16 * 8
    base = estimate_step5_flops(cfg, 3, [3], 1)
    actual = estimate_step5_flops(cfg, 3, [3], 1, step5_image_shapes=[(8, 8), (8, 8)])
    assert (actual - base) * 1e12 == pytest.approx(2 * expected)


@pytest.mark.parametrize(
    ("freeze_encoder", "freeze_projector", "expected_per_image"),
    [
        (False, False, 42240),
        (False, True, 41984),
        (True, False, 14848),
        (True, True, 14592),
    ],
)
def test_vision_freeze_flags_follow_gradient_requirements(freeze_encoder, freeze_projector, expected_per_image):
    cfg = SimpleNamespace(
        model_type="step5_vl",
        text_config=config(),
        freeze_vision_encoder=freeze_encoder,
        freeze_vision_projector=freeze_projector,
        vision_config=SimpleNamespace(
            width=4,
            patch_size=2,
            mlp_ratio=2,
            num_channels=3,
            layers=1,
            use_cls_token=False,
        ),
    )
    # Forward-only: patch=1536, ViT=8192, downsample convolutions=4608,
    # projector=256 FLOPs. Downsampling belongs to the encoder.
    # The patch has no input gradient. The projector needs dInput iff the
    # encoder trains, and dWeight iff the projector trains, independently.
    base = estimate_step5_flops(cfg, 3, [3], 1)
    actual = estimate_step5_flops(cfg, 3, [3], 1, step5_image_shapes=[(8, 8)] * 2)
    assert (actual - base) * 1e12 == pytest.approx(2 * expected_per_image)


@pytest.mark.parametrize("as_dict", [False, True])
def test_aux_disabled_preserves_eager_indexer_scoring(as_dict):
    sparse = dict(
        enabled=True,
        proxy_dim=2,
        sparse_indexer_num_heads=2,
        region_block_size=2,
        topk=1,
        apply_to_layer_types=["full_attention"],
    )
    cfg = config(
        sparse_config=sparse if as_dict else SimpleNamespace(**sparse),
        enable_dsa_aux_loss=False,
    )
    # With eager selection, turning aux off removes only the detached teacher:
    # indexer projections and all four visible-region scores still execute.
    expected = (6 * 608 + 2 * 80) * 5 + 12 * 8 * 13 + 2 * 2 * 2 * 4
    actual = estimate_step5_flops(cfg, 5, [5], 1) * 1e12
    assert actual == pytest.approx(expected)
    cfg.enable_dsa_aux_loss = True
    with_teacher = estimate_step5_flops(cfg, 5, [5], 1) * 1e12
    assert with_teacher - actual == pytest.approx(2 * 8 * 2 * 4)


@pytest.mark.parametrize(
    ("block", "topk", "lengths"),
    [
        (2, 1, [0, 1, 4]),
        (2, 1, [4, 5, 7]),
        (8, 512, [4103]),
        (8, 512, [4104]),
        (8, 512, [4105]),
        (8, 512, [4104, 4105, 4112]),
    ],
)
def test_gbsa_aux_disabled_scores_only_sparse_tail(block, topk, lengths):
    cfg = config(
        enable_dsa_aux_loss=False,
        use_gbsa_full_sequence=True,
        sparse_config=SimpleNamespace(
            enabled=True,
            proxy_dim=2,
            sparse_indexer_num_heads=2,
            region_block_size=block,
            topk=topk,
            apply_to_layer_types=["full_attention"],
        ),
    )
    tokens = sum(lengths)
    # Independent query-by-query reference: complete selected regions plus
    # the causal part of the current region. Prefix queries need no scoring.
    attention_pairs = sum(
        min(q // block, topk) * block + q % block + 1
        for length in lengths
        for q in range(length)
    )
    scored_regions = sum(
        q // block for length in lengths for q in range((topk + 1) * block, length)
    )
    expected = (6 * 608 + 2 * 80) * tokens + 12 * 8 * attention_pairs + 2 * 2 * 2 * scored_regions
    actual = estimate_step5_flops(cfg, tokens, lengths, 1) * 1e12
    assert actual == pytest.approx(expected, rel=1e-12)


def test_image_shapes_no_pixel_reads_and_precomputed_embeddings():
    overview = torch.empty((2, 1, 3, 728, 728), device="meta")
    crops = torch.empty((3, 3, 504, 504), device="meta")
    inputs = [
        None,
        NonTensorData({"pixel_values": overview, "patch_pixel_values": crops}),
        {"image_embeds": torch.empty((4, 8), device="meta")},
        {"patch_pixel_values": torch.empty((0, 3, 504, 504), device="meta")},
    ]
    assert step5_image_shapes(inputs) == [(728, 728)] * 2 + [(504, 504)] * 3


def test_counter_registration_scaling_and_actor_only(monkeypatch):
    monkeypatch.setattr("verl.utils.flops_counter.get_device_flops", lambda: 432)
    counter = FlopsCounter(config())
    rate, peak = counter.estimate_flops([3, 3], 2)
    assert peak == 432
    assert rate == pytest.approx(estimate_step5_flops(config(), 3, [3], 1))
    assert counter.estimate_flops([3], 1, forward_only=True)[0] == 0
    assert get_device_flops(device_name="Ascend950DT") == 432


def test_invalid_time_and_empty_batch():
    with pytest.raises(ValueError, match="elapsed"):
        estimate_step5_flops(config(), 3, [3], 0)
    assert estimate_step5_flops(config(), 0, [], 1) == 0


def test_training_worker_mfu_metadata_reduction_and_logging(monkeypatch, capsys):
    import inspect
    from contextlib import nullcontext
    from types import MethodType

    from verl.utils import tensordict_utils as tu
    from verl.utils.tracking import Tracking
    from verl.workers import engine_workers as workers

    cfg = SimpleNamespace(
        model_type="step5_vl",
        text_config=config(
            enable_dsa_aux_loss=False,
            use_gbsa_full_sequence=True,
            sparse_config=SimpleNamespace(
                enabled=True,
                proxy_dim=2,
                sparse_indexer_num_heads=2,
                region_block_size=2,
                topk=1,
                apply_to_layer_types=["full_attention"],
            ),
        ),
        freeze_vision_encoder=True,
        freeze_vision_projector=True,
        vision_config=SimpleNamespace(
            width=4,
            patch_size=2,
            mlp_ratio=2,
            num_channels=3,
            layers=1,
            use_cls_token=False,
        ),
    )
    sync_calls = []
    device = SimpleNamespace(
        synchronize=lambda: sync_calls.append(True), max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0
    )
    monkeypatch.setattr(workers, "get_torch_device", lambda: device)
    monkeypatch.setattr("verl.utils.flops_counter.get_device_flops", lambda: 432)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda out, value, group: out.__setitem__(slice(None), [value, value])
    )
    monkeypatch.setattr(workers, "allgather_dict_into_dict", lambda data, group: data)

    def all_reduce(tensor, op, group=None):
        if op == torch.distributed.ReduceOp.MAX:
            assert tensor.dtype == torch.float32  # NPU does not support float64.
            tensor.fill_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    engine = SimpleNamespace(
        get_data_parallel_group=lambda: "dp",
        train_mode=lambda **kw: nullcontext(),
        train_batch=lambda *a, **kw: {"metrics": {}, "loss": [0.0], "model_output": {}},
        is_mp_src_rank_with_outputs=lambda: True,
    )
    worker = SimpleNamespace(
        engine=engine,
        device_name="cpu",
        loss_fn=lambda: None,
        model_config=SimpleNamespace(hf_config=cfg, get=lambda key, default: default),
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=8,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
        flops_counter=FlopsCounter(cfg),
    )
    worker._postprocess_output = MethodType(workers.TrainingWorker._postprocess_output, worker)
    data = tu.get_tensordict(
        tensor_dict={},
        non_tensor_dict={
            "global_token_num": [3, 3],
            "multi_modal_inputs": [{"pixel_values": torch.empty((1, 3, 8, 8))}],
        },
    )
    result = inspect.unwrap(workers.TrainingWorker.train_batch)(worker, data)
    mfu = tu.get(result, "metrics")["mfu"]
    # Two global sequences, each 12000 text + 14592 frozen-vision FLOPs,
    # divided by max-rank elapsed time (2 s), device peak, and two devices.
    expected = 2 * (12000 + 14592) / 2 / 1e12 / 432 / 2
    assert mfu == pytest.approx(expected, rel=1e-10, abs=0)
    assert sync_calls == [True]
    tracker = Tracking(project_name="test", experiment_name="actor-mfu", default_backend="console")
    tracker.log(data={"perf/mfu/actor": mfu}, step=1)
    assert "perf/mfu/actor" in capsys.readouterr().out

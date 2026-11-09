"""Cross-package CPU checks for the Step5 rollout wire format and packed actor."""
import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

consumer = pytest.importorskip("fsdp_turbo.utils.region_replay")
producer = pytest.importorskip("vllm_ascend.models.step5.region_replay")
SINGLE = SimpleNamespace(world_size=1, rank_in_group=0)


def frame(start, ids, block=8, layer=11):
    groups, rows, topk = ids.shape
    cap = producer.RegionCapture(["req"], [start], [rows])
    cap.record(f"model.layers.{layer}.attn", ids, 0, rows, groups, block, topk)
    pending = cap.finish(SINGLE)
    return pending().get("req", b"") if pending else b""


@pytest.mark.parametrize("position", [4104, 5122, 65536, 524288, 524304, 2**24])
def test_width_transitions(position):
    topk = 512
    last = position // 8 - 1
    ids = torch.arange(last - topk + 1, last + 1, dtype=torch.int32).reshape(1, 1, -1)
    data = frame(position, ids)
    assert len(data) <= 36 + 4 * topk
    [(start, actual)] = consumer.RegionReplay(data).segments(11, position + 1, 1, topk, 8)
    assert start == position and torch.equal(actual, ids)


def test_mixed_requests_split_decode_prefill_and_layers():
    cap = producer.RegionCapture(["decode", "short", "prefill"], [25, 0, 22], [1, 2, 4])
    decode = torch.tensor([[[0, 2]]], dtype=torch.int32)
    prefill = torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1], [1, 2], [0, 2]]], dtype=torch.int32)
    for layer in (3, 11):
        name = f"model.layers.{layer}.attn"
        cap.record(name, decode, 0, 1, 1, 8, 2)
        cap.record(name, prefill, 1, 6, 1, 8, 2)
    records = cap.finish(SINGLE)()
    assert records.keys() == {"decode", "prefill"}
    for layer in (3, 11):
        [(start, ids)] = consumer.RegionReplay(records["prefill"]).segments(layer, 25, 1, 2, 8)
        assert start == 24 and torch.equal(ids, prefill[:, 4:5])
        assert not list(consumer.RegionReplay(records["prefill"]).segments(layer, 24, 1, 2, 8))
        [(start, ids)] = consumer.RegionReplay(records["decode"]).segments(layer, 100, 1, 2, 8)
        assert start == 25 and torch.equal(ids, decode)


def test_resumed_request_overlap_and_gap():
    a = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int32)
    b = torch.tensor([[[0, 2], [0, 2]]], dtype=torch.int32)
    payload = frame(24, a) + frame(25, b) + frame(30, a)
    ranges = list(consumer.RegionReplay(payload).segments(11, 31, 1, 2, 8))
    assert [i for i, _ in ranges] == [24, 30]
    assert torch.equal(ranges[0][1], torch.cat((a, b[:, 1:]), dim=1))
    assert torch.equal(ranges[1][1], a[:, :1])
    with pytest.raises(ValueError):
        consumer.RegionReplay(payload[:-1])


def test_tp_rank_order():
    ids = torch.tensor([[[0, 1]]], dtype=torch.int32)
    cap = producer.RegionCapture(["req"], [24], [1])
    cap.record("model.layers.11.attn", ids, 0, 1, 1, 8, 2)
    tp = SimpleNamespace(rank_in_group=0, world_size=2,
                         all_gather=lambda data, dim: torch.cat((data, torch.tensor([6], dtype=torch.uint8))))
    [(start, result)] = consumer.RegionReplay(cap.finish(tp)()["req"]).segments(11, 25, 2, 2, 8)
    assert start == 24 and result.tolist() == [[[0, 1]], [[1, 2]]]


def test_reference_dump():
    """Run against the original full layer-11 evidence when it is available."""
    base = Path("/tmp/step5-dsa11-a204/infer")
    if not base.exists():
        pytest.skip("A5-02-04 reference dump is unavailable")
    parts = [torch.load(base / str(rank) / "ids.pt", mmap=True, weights_only=True) for rank in range(4)]
    ids = torch.stack(parts)
    rows, topk = ids.shape[1:]
    data = frame(0, ids)
    [(start, actual)] = consumer.RegionReplay(data).segments(11, rows, 4, topk, 8)
    assert torch.equal(actual, ids[:, start:])
    assert len(data) - struct.calcsize("<4s8I") == 326080

def test_tnd_and_actor_transport(monkeypatch):
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict
    from fsdp_turbo.models.step5 import gbsa_qsag, modeling_step5 as model
    from verl.models.fsdp_turbo.step5 import Step5ModelAdapter

    # Two real documents followed by engine alignment padding. The first query
    # of doc 2 must not inherit doc 1's absolute position or replay selection.
    a = frame(24, torch.tensor([[[1, 2], [0, 2]]]))
    tokens = [torch.arange(27), torch.arange(9)]
    td = list_of_dict_to_tensordict([{"input_ids": x, "routed_experts": torch.zeros((len(x), 1, 1), dtype=torch.long), "step5_region_frames": payload}
                                     for x, payload in zip(tokens, (a, b""))])
    engine = SimpleNamespace(use_remove_padding=True, engine_config=SimpleNamespace(
        router_replay=SimpleNamespace(mode="R3")))
    inputs = Step5ModelAdapter().prepare_model_inputs(engine, td, {"input_ids": torch.zeros(1, 38)})
    assert inputs["actual_seq_len"].tolist() == [27, 36, 38]
    docs = inputs["region_replay"]
    config = SimpleNamespace(sparse_config=SimpleNamespace(region_block_size=8, topk=2),
        use_gbsa_full_sequence=True, enable_dsa_aux_loss=False, dsa_aux_query_chunk_size=4,
        use_indexer_region_score=False, use_fp8_indexer_selection=False,
        use_fp32_indexer_scores=True, enable_region_topk_ids=False)
    x = torch.ones(1, 38, 1, 2, requires_grad=True)
    att = SimpleNamespace(layer_idx=11, config=config, num_kv_heads=1,
                          _indexer_projections=lambda *args: (x, x, x, x),
                          _region_summaries=lambda *args: x)
    calls = []
    def native(q, summaries, weights, start, block, topk, **kwargs):
        calls.append((start, len(q)))
        return torch.tensor([[[0, 1, -1]]]), torch.tensor([[2]])
    def core(q, k, v, ids, counts, backward_ids, backward_counts, cu_q, cu_k, **kw):
        assert ids[:, 24:27].tolist() == [[[1, 2, -1], [0, 2, -1], [0, 1, -1]]]
        assert torch.equal(backward_ids[0, :, :27], ids[:, :27])
        assert torch.equal(backward_counts[0, :, :27], counts[:, :27])
        assert counts[0, 27] == 0 and counts[0, 36] == 0
        return q * 2
    monkeypatch.setattr(model, "_step5_gbsa_select_long_tail_regions", native)
    monkeypatch.setattr(gbsa_qsag, "gbsa_qsag_batched", core)
    def forward(q):
        return model.Step5Attention._tnd_gbsa_attention(att, q, q, q, q, None, (27, 36, 38), docs)
    from torch.utils.checkpoint import DefaultDeviceType, checkpoint
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    # Nonlinear objective forces autograd to use the recomputed forward.
    checkpoint(lambda q: forward(q).square(), x, use_reentrant=True).sum().backward()
    assert calls == [(26, 1), (26, 1)] and torch.equal(x.grad, 8 * x.detach())

def test_empty_capture_and_layer_coverage():
    cap = producer.RegionCapture(["short", "edge"], [0, 22], [8, 2], 24, {11})
    assert cap.finish(SINGLE) is None
    cap = producer.RegionCapture(["long"], [23], [2], 24, {11})
    with pytest.raises(RuntimeError, match="Missing"):
        cap.finish(SINGLE)
    ids = torch.tensor([[[0, 1], [0, 2]]], dtype=torch.int32)
    cap.record("model.layers.11.attn", ids, 0, 2, 1, 8, 2)
    assert cap.finish(SINGLE)()["long"] == frame(24, ids[:, 1:])


@pytest.mark.parametrize("position", [24, 128, 524304])
def test_graph_decode_frames_match_eager_after_reordering(position):
    layers, block, topk = (3, 11), 8, 2
    # The graph bucket has four rows, but only three real requests. Two local
    # KV groups expose stride/layout bugs when excluding padding.
    buf = torch.full((2, 2, 4, topk), -1, dtype=torch.int32)
    group = SimpleNamespace(world_size=2, rank_in_group=0,
                            all_gather=lambda t, dim: torch.cat((t, t), dim=dim))
    for names, starts in ((["a", "b", "c"], [23, position, 25]),
                          (["c", "a", "b"], [26, 24, position + 1])):
        cap = producer.RegionCapture(names, starts, [1] * 3, 24, set(layers))
        for i, layer in enumerate(layers):
            for row, start in enumerate(starts):
                last = start // block - 1
                buf[i, :, row, 0] = (i + row) % last
                buf[i, :, row, 1] = last
            cap.record(f"model.layers.{layer}.attn", buf[i, :, :3], 0, 3, 2, block, topk)
        expected = cap.finish(group)()
        pending = cap.finish_decode(buf, layers, block, group)
        buf.fill_(-1)  # next graph can overwrite the staging buffer before host consumption
        assert pending() == expected
    cap = producer.RegionCapture(["a"], [23], [1], 24, set(layers))
    assert cap.finish_decode(buf, layers, block, group) is None


def test_rollout_payload_through_engine_and_agent():
    import msgspec
    from vllm.v1.engine import EngineCoreOutput, FinishReason
    from vllm.v1.engine.output_processor import RequestState
    from vllm.sampling_params import RequestOutputKind
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    payload = frame(24, torch.tensor([[[1, 2]]]))
    core = EngineCoreOutput(request_id="sample", new_token_ids=[1], finish_reason=FinishReason.STOP,
                            step5_region_frames=payload)
    # EngineCoreOutput is array-like; the byte field must survive the actual wire layout.
    encoded = msgspec.msgpack.decode(msgspec.msgpack.encode(core))
    received = dict(zip(EngineCoreOutput.__struct_fields__, encoded))["step5_region_frames"]
    state = SimpleNamespace(output_kind=RequestOutputKind.DELTA, request_index=0,
                            detokenizer=SimpleNamespace(get_next_output_text=lambda *args: ""),
                            logprobs_processor=SimpleNamespace(logprobs=None, cumulative_logprob=None),
                            routed_experts_chunks=[], step5_region_chunks=[received])
    result = RequestState._new_completion_output(state, [1], FinishReason.STOP, None)
    agent = AgentLoopOutput(prompt_ids=[1, 2], response_ids=[3], response_mask=[1],
                            metrics=AgentLoopMetrics(), extra_fields={"step5_region_frames": result.step5_region_frames})
    fields = agent.as_dict()
    assert fields["step5_region_frames"] == payload
    assert "step5_region_frames" not in fields["extra_fields"]

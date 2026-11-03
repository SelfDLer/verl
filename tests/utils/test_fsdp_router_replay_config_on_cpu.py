"""FSDP replay overrides must survive the production Hydra/dataclass conversion."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

import verl
from verl.models.fsdp_turbo.step5 import Step5ModelAdapter
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.engine import EngineRouterReplayConfig


@pytest.mark.parametrize("strategy", ["fsdp2", "fsdp_turbo"])
@pytest.mark.parametrize("mode", ["disabled", "R3"])
def test_fsdp_replay_override_is_typed(strategy, mode):
    config_dir = str(Path(verl.__file__).parent / "trainer/config")
    overrides = [
        f"actor_rollout_ref.actor.strategy={strategy}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    ]
    if mode == "R3":
        overrides += [
            "++actor_rollout_ref.actor.fsdp_config.router_replay.mode=R3",
            "actor_rollout_ref.rollout.enable_rollout_routing_replay=true",
        ]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.engine is actor.fsdp_config
    assert isinstance(actor.engine.router_replay, EngineRouterReplayConfig)
    assert actor.engine.router_replay.mode == mode
    assert config.actor_rollout_ref.rollout.enable_rollout_routing_replay == (mode == "R3")

    if strategy == "fsdp_turbo":
        engine = SimpleNamespace(engine_config=actor.engine)
        tokens = torch.ones(1, 4, dtype=torch.long)
        routes = torch.zeros(1, 4, 2, 2, dtype=torch.long)
        model_inputs = {"input_ids": tokens}
        adapter = Step5ModelAdapter()
        if mode == "R3":
            with pytest.raises(ValueError, match="requires routed_experts"):
                adapter.prepare_model_inputs(engine, {}, dict(model_inputs))
        output = adapter.prepare_model_inputs(
            engine,
            {"routed_experts": routes, "response_mask": torch.tensor([[0, 0, 1, 1]])},
            model_inputs,
        )
        if mode == "R3":
            assert output["routed_experts"] is routes
            assert output["routing_replay_sample_mask"].tolist() == [True]
        else:
            assert "routed_experts" not in output

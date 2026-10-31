# Copyright 2025 Individual Contributor: TomQunChaoA
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

import math
import unittest

import pytest
import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics


class TestMetrics(unittest.TestCase):
    def test_calculate_debug_metrics(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor(
                    [
                        [-1.5085, -0.1200, -0.6650, -0.4823, -0.1426, -1.5557, -2.8532, -0.3919, -0.4294, -0.4700],
                        [-0.0585, -0.0573, -0.4681, -0.5187, -0.7451, -1.2737, -0.0682, -0.4284, -0.5754, -0.0611],
                    ]
                ),
                "old_log_probs": torch.tensor(
                    [
                        [-1.8636, -0.7863, -0.2136, -0.4376, -2.0257, -0.2579, -1.1547, -0.5203, -0.3802, -0.9872],
                        [-0.3507, -0.5426, -0.2725, -0.4637, -0.3577, -0.3733, -1.7560, -1.9542, -0.4229, -1.3098],
                    ]
                ),
                "loss_mask": torch.tensor([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]),
                "responses": torch.zeros((2, 10)),
            }
        )
        metrics = calculate_debug_metrics(data)
        print(metrics)
        assert metrics["training/rollout_probs_diff_valid"] == 1


@pytest.mark.parametrize("mask_kind", ["response_mask", "attention_mask", "none"])
def test_log_prob_diff_uses_same_tokens_as_prob_diff(mask_kind):
    actor = torch.tensor([[-1.0, -3.0, -1000.0]])
    rollout = torch.tensor([[-2.0, -6.0, -1100.0]])
    tensors = {
        "old_log_probs": actor,
        "rollout_log_probs": rollout,
        "responses": torch.zeros((1, 3), dtype=torch.long),
    }
    if mask_kind == "response_mask":
        tensors["response_mask"] = torch.tensor([[1, 1, 0]])
        tensors["attention_mask"] = torch.ones((1, 5))
    elif mask_kind == "attention_mask":
        tensors["attention_mask"] = torch.tensor([[1, 1, 1, 1, 0]])
    selected = slice(None) if mask_kind == "none" else slice(0, 2)
    expected_log_diff = torch.tensor([1.0, 3.0, 100.0])[selected]
    expected_prob_diff = (actor.exp() - rollout.exp()).abs()[0, selected]

    metrics = calculate_debug_metrics(DataProto.from_dict(tensors))
    assert metrics["training/rollout_probs_diff_valid"] == 1
    for prefix, expected in [("log_probs", expected_log_diff), ("probs", expected_prob_diff)]:
        for statistic in ("max", "mean", "std"):
            assert metrics[f"training/rollout_{prefix}_diff_{statistic}"] == pytest.approx(
                getattr(expected, statistic)().item()
            )


@pytest.mark.parametrize("mask", [[[0, 0]], [[1, 0]]])
def test_log_prob_diff_empty_and_single_token(mask):
    metrics = calculate_debug_metrics(
        DataProto.from_dict(
            {
                "old_log_probs": torch.tensor([[-1.0, float("nan")]]),
                "rollout_log_probs": torch.tensor([[-3.0, float("nan")]]),
                "response_mask": torch.tensor(mask),
                "responses": torch.zeros((1, 2), dtype=torch.long),
            }
        )
    )
    valid = mask[0][0]
    assert metrics["training/rollout_probs_diff_valid"] == valid
    for statistic in ("max", "mean", "std"):
        actual = metrics[f"training/rollout_log_probs_diff_{statistic}"]
        if not valid or statistic == "std":
            assert math.isnan(actual)
        else:
            assert actual == 2.0


def test_log_prob_diff_console_logging(capsys):
    from verl.utils.tracking import Tracking

    data = DataProto.from_dict(
        {
            "old_log_probs": torch.tensor([[-1.0, -3.0]]),
            "rollout_log_probs": torch.tensor([[-2.0, -6.0]]),
            "response_mask": torch.ones((1, 2)),
            "responses": torch.zeros((1, 2), dtype=torch.long),
        }
    )
    tracker = Tracking(project_name="test", experiment_name="log-prob-diff", default_backend="console")
    tracker.log(data=calculate_debug_metrics(data), step=1)
    output = capsys.readouterr().out
    for prefix in ("probs", "log_probs"):
        for statistic in ("max", "mean", "std"):
            assert f"training/rollout_{prefix}_diff_{statistic}" in output


if __name__ == "__main__":
    unittest.main()

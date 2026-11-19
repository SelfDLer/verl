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

"""Model configuration for the optional FSDPTurbo Step5 integration."""

from verl.workers.config import HFModelConfig


class Step5HFModelConfig(HFModelConfig):
    """Load the optional Step5 processor provided by FSDPTurbo.

    Select it with ``actor_rollout_ref.model._target_=verl.models.transformers.step5.Step5HFModelConfig``.
    """

    def __post_init__(self) -> None:
        from fsdp_turbo.models.step5 import Step5Processor, Step5VLConfig, register_step5
        from transformers import AutoProcessor

        # vLLM can register its own Step5 config between two verl initializations.
        register_step5()
        AutoProcessor.register(Step5VLConfig, Step5Processor, exist_ok=True)
        super().__post_init__()

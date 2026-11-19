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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import FSDPEngineConfig, HFModelConfig

if TYPE_CHECKING:
    from fsdp_turbo.fsdp_turbo_config import DistributedConfig, FSDPTurboConfig
    from torch import nn


class Step5HFModelConfig(HFModelConfig):
    """Load the Step5 processor and training model provided by FSDPTurbo.

    Select it with ``actor_rollout_ref.model._target_=verl.models.transformers.step5.Step5HFModelConfig``.
    """

    def __post_init__(self) -> None:
        from fsdp_turbo.models.step5 import Step5Processor, Step5VLConfig, register_step5
        from transformers import AutoProcessor

        # vLLM can register its own Step5 config between two verl initializations.
        register_step5()
        AutoProcessor.register(Step5VLConfig, Step5Processor, exist_ok=True)
        super().__post_init__()

    @property
    def fsdp_turbo_config_class(self) -> type[FSDPTurboConfig]:
        """Use the Step5 sharding precision policies when validating the Turbo config."""
        from fsdp_turbo.models.step5.fsdp import Step5FSDPTurboConfig

        return Step5FSDPTurboConfig

    def build_fsdp_turbo_model(
        self,
        engine_config: FSDPEngineConfig,
        distributed_config: DistributedConfig,
        *,
        init_device: Literal["cpu", "meta"],
    ) -> nn.Module:
        """Build mixed-precision Step5 parameters before the engine captures its unsharded state."""
        from fsdp_turbo.models.step5.fsdp import build_step5_fsdp_model

        if (
            self.use_remove_padding
            or engine_config.ulysses_sequence_parallel_size != 1
            or distributed_config.ulysses_parallel_size != 1
        ):
            raise ValueError("Step5 FSDPTurbo requires padded inputs and Ulysses sequence parallel size 1")
        if self.use_fused_kernels or self.use_liger:
            raise ValueError(
                "Step5 FSDPTurbo requires the standard logits forward (disable use_fused_kernels and use_liger)"
            )
        dtype = engine_config.model_dtype
        if dtype is None:
            dtype = "bf16" if engine_config.forward_only else "fp32"
        model = build_step5_fsdp_model(
            self.local_path,
            self.hf_config,
            distributed_config,
            dtype=PrecisionType.to_dtype(dtype),
            init_device=init_device,
        )
        if self.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model

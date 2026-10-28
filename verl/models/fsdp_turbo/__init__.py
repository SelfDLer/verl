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

"""Model-specific construction and checkpoint layout for the FSDP-Turbo engine.

Unknown model types retain the engine's existing Hugging Face loading path.
Optional model implementations are imported only when selected.
"""

import importlib


class ModelAdapter:
    """Default no-op hooks preserving the existing engine path."""

    def build_module(self, engine, default_builder):
        """Construct the model using the engine default."""
        return default_builder()

    def prepare_fsdp(self, engine, module):
        """Apply model-specific preparation before sharding, when needed."""
        pass

    def export_weights(self, weights):
        """Preserve the original lazy checkpoint iterator."""
        return weights

    def prepare_model_inputs(self, engine, micro_batch, model_inputs):
        """Attach optional model-specific inputs after the engine applies padding."""
        return model_inputs


_MODEL_ADAPTERS = {
    "step5_vl": "verl.models.fsdp_turbo.step5.Step5ModelAdapter",
}


def get_model_adapter(model_type):
    """Resolve one model adapter without importing unrelated optional dependencies."""
    path = _MODEL_ADAPTERS.get(model_type)
    if path is None:
        return ModelAdapter()
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)()

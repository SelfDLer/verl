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

"""Bounded-lifetime staging for loaders requiring a complete weight update.

Enable with rollout.checkpoint_engine.weight_update_staging = "cpu" or "device".
CPU staging requires host RAM for a full checkpoint per rollout worker; device
staging requires equivalent accelerator memory. Neither mode is enabled by default.
"""


class WeightUpdateBuffer:
    """Own transport tensors until the complete update can be loaded once."""

    def __init__(self, staging):
        if staging not in ("cpu", "device"):
            raise ValueError("weight_update_staging must be 'cpu' or 'device'")
        self.staging = staging
        self._weights = []

    def receive(self, weights, is_last, load_weights):
        """Copy a reused bucket and invoke the loader only on the final bucket."""
        # Transport buckets are reused: retaining views corrupts earlier tensors.
        self._weights.extend(
            (name, tensor.to(device="cpu", copy=True) if self.staging == "cpu" else tensor.clone())
            for name, tensor in weights
        )
        if is_last:
            try:
                load_weights(self._weights)
            finally:
                self.clear()

    def clear(self):
        """Release staged tensors after success or transport/loader failure."""
        self._weights.clear()

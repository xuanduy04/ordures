# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Second example launcher: mixed MATH + MCQ, using normal examples/run_grpo.py.

What this file does:
1) Register the custom MCQ environment name used by this second example.
2) Patch env-name extraction to support list-based multi-dataset configs.
3) Call the stock examples.run_grpo.main() (no custom train loop here).

Why the tiny patch exists:
- The stock env-name extraction helper expects dict-style splits and can miss
  env names when `data.train` / `data.validation` are YAML lists.
- This example uses list-based multi-task datasets, so we provide a robust
  extractor here while keeping the rest of the pipeline unchanged.
"""

from __future__ import annotations

from typing import Any

from nemo_rl.environments.utils import register_env
import nemo_rl.data.utils as data_utils


def _extract_env_names_multidataset(data_config: dict[str, Any]) -> list[str]:
    """Extract unique env names from train/validation/default for dict-or-list splits."""
    env_names: set[str] = set()

    for split_key in ("train", "validation", "default"):
        split_cfg = data_config.get(split_key)
        if split_cfg is None:
            continue

        # Normalize to list so both single-dataset and multi-dataset forms work.
        split_entries = split_cfg if isinstance(split_cfg, list) else [split_cfg]

        for entry in split_entries:
            if not isinstance(entry, dict):
                continue
            if "env_name" in entry and entry["env_name"] is not None:
                env_names.add(str(entry["env_name"]))

    return list(env_names)


# Patch the helper used by setup_response_data in nemo_rl.data.utils.
data_utils.extract_necessary_env_names = _extract_env_names_multidataset

# Register a separate env name for this second multitask example.
register_env(
    env_name="mcq_exact",
    actor_class_fqn="examples.mcq_exact_env.MCQExactMatchEnvironment",
)

# Reuse the normal GRPO entrypoint exactly as requested.
from examples.run_grpo import main


if __name__ == "__main__":
    main()

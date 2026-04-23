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

"""Custom minimal MCQ environment used by GRPO.

This file intentionally includes dense comments so each moving piece is explicit.
The environment gives a reward of 1.0 when:
    normalized(model_answer) == normalized(ground_truth)
where normalize(x) = x.strip().lower()

Everything else gets reward 0.0.
"""

from typing import Any, TypedDict

import ray
import torch

# LLMMessageLogType is the conversation structure used by NeMo RL.
# It is effectively:
#   list[dict[str, Union[str, torch.Tensor]]]
# where each dict is a turn like {"role": "user"|"assistant"|"environment", "content": "..."}.
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class MCQMetadata(TypedDict):
    # This is the metadata payload we expect per sample from the data processor.
    # For this minimal env, we only need one value: the expected answer label/text.
    # Example:
    #   {"ground_truth": "b"}
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MCQExactMatchEnvironment(EnvironmentInterface[MCQMetadata]):
    """Single-turn MCQ environment with exact-match rewards.

    Why "single-turn":
    - We always mark each sample as terminated in one step.
    - The model responds once per prompt and we score immediately.
    """

    def __init__(self, cfg: dict[str, Any]):
        # Keep config around for future extension (timeouts, parsing mode, etc).
        # For now, this minimal env does not require any config keys.
        self.cfg = cfg

    @staticmethod
    def _normalize_answer(text: str) -> str:
        # The exact normalization requested by you:
        # - strip() removes leading/trailing whitespace
        # - lower() makes comparison case-insensitive
        return text.strip().lower()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MCQMetadata],
    ) -> EnvironmentReturn[MCQMetadata]:
        # We build batched return fields progressively.
        rewards: list[float] = []
        observations: list[dict[str, str]] = []

        # Iterate batch-wise over conversations and their metadata.
        # zip(...) is safe because NeMo RL provides matched batch lengths.
        for conversation, env_info in zip(message_log_batch, metadata):
            # Extract the assistant output from the full conversation.
            # We concatenate all assistant turns so this still works if more than one
            # assistant turn exists in a conversation.
            assistant_response = "".join(
                str(turn["content"])
                for turn in conversation
                if turn.get("role") == "assistant"
            )
            # Apply required normalization to both prediction and target.
            predicted = self._normalize_answer(assistant_response)
            expected = self._normalize_answer(str(env_info["ground_truth"]))
            is_correct = predicted == expected

            # Reward shape is scalar float per sample.
            rewards.append(1.0 if is_correct else 0.0)
            # Observation is optional but useful for debug logging.
            observations.append(
                {
                    "role": "environment",
                    "content": "Environment: correct"
                    if is_correct
                    else "Environment: incorrect",
                }
            )

        # Convert Python list -> CPU tensor because the environment contract expects
        # tensor rewards in batched form.
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        # Single-turn environment:
        # terminated = 1 for every sample in this batch.
        # We match reward tensor shape for convenience.
        terminated_tensor = torch.ones_like(rewards_tensor).cpu()

        return EnvironmentReturn(
            # Environment text feedback for each sample.
            observations=observations,
            # Pass metadata through unchanged.
            metadata=metadata,
            # No dynamic stop strings for the next turn since we terminate immediately.
            next_stop_strings=[None] * len(message_log_batch),
            # Batched scalar rewards.
            rewards=rewards_tensor,
            # Batched done flags (all done after one step).
            terminateds=terminated_tensor,
            # No extracted answer payload needed for this minimal example.
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        # Emit an easy-to-read metric at env level:
        # average reward == exact-match accuracy for this binary reward.
        metrics = {"accuracy": float(batch["rewards"].float().mean().item())}
        # Return batch unchanged + computed metrics.
        return batch, metrics

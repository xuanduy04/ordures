"""Register a truncating variant of math_hf_data_processor."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.data.processors import (
    PROCESSOR_REGISTRY,
    register_processor,
)

TokenizerType = PreTrainedTokenizerBase


PROCESSOR_NAME = "math_hf_data_processor_truncate_last"


def math_hf_data_processor_truncate_last(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary into a DatumSpec for the Reward Model Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []
    formatted_content = (
        task_data_spec.prompt.format(problem) if task_data_spec.prompt else problem
    )
    user_message = {
        "role": "user",
        "content": formatted_content,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller by keeping the last max_seq_length tokens
        # but keep sample active instead of dropping the sample like `nemo_rl.data.processors.math_hf_data_processor`
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][-max_seq_length:]
        length = sum(len(m["token_ids"]) for m in message_log)

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


if PROCESSOR_NAME not in PROCESSOR_REGISTRY:
    register_processor(PROCESSOR_NAME, math_hf_data_processor_truncate_last)

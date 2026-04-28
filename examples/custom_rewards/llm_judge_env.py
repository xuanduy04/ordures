from __future__ import annotations

import re
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlparse

import ray
import requests
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from examples.custom_rewards.format_reward import verify_format


class LLMJudgeMetadata(TypedDict):
    # `math_hf_data_processor` writes the expected answer under this key.
    # We reuse the same schema so this environment can drop into existing examples.
    ground_truth: str


class LLMJudgeEnvConfig(TypedDict):
    # Base vLLM server URL (just host:port style is valid too).
    vllm_server_url: str
    # Judge model passed as `model` in the JSON payload.
    model: str
    # Judge behavior/rubric prompt.
    system_prompt: str
    # Number of Ray workers to judge in parallel. Defaults to 1.
    num_workers: NotRequired[int]

    max_judge_output_tokens: int

# if VOID_ANSWER_TOKEN in judge prompt, then ignore the judge and return 0.0 reward
VOID_ANSWER_TOKEN = "<|void_answer_VGhpcyBsaWJyYXJ5IGlzIHNvIHRyYXNoIQ==|>"
DEFAULT_JUDGE_SYSTEM_PROMPT = r"""You are an impartial evaluator.
You are given:
- A gold (correct) label
- A model-generated answer

Your task is to determine whether the answer is factually consistent with the gold label.
Rules:
- Return 1 inside \boxed{}, i.e. \boxed{1} if the answer conveys the same meaning as the gold label.
- Otherwise, return 0 inside \boxed{}, i.e. \boxed{0} if the answer contradicts, is inconsistent, incomplete, or adds incorrect information.
- Minor wording differences are acceptable if the meaning is the same.
- Do not use external knowledge, only compare the answer to the gold label.
- Accept the answer regardless of formatting or structural differences, as long as the core factual content aligns with the gold label.
- If the model-generated answer contains a clear final choice (e.g., "A", "B", etc.), prioritize that choice over any additional text, even if extra, redundant, or noisy content appears. For example, "C. Not wrong, Wrong" should be interpreted as the model selecting C."""


def _normalize_to_chat_completions_url(raw_url: str) -> str:
    """Normalize server URL and produce the /v1/chat/completions endpoint URL.

    Accepts:
    - localhost:8000
    - http://localhost:8000
    - http://localhost:8000/v1/chat/completions
    """
    normalized = raw_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("llm_judge config requires non-empty `vllm_server_url`.")

    # Allow user to provide plain host:port without scheme.
    if "://" not in normalized:
        normalized = f"http://{normalized}"

    parsed_url = urlparse(normalized)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(
            "Expected `vllm_server_url` like 'localhost:8000' or "
            "'http://localhost:8000'."
        )

    if normalized.endswith("/v1/chat/completions"):
        return normalized

    return f"{normalized}/v1/chat/completions"


def _extract_role_content(conversation: LLMMessageLogType, role: str) -> str:
    # Message logs are lists of role/content dictionaries.
    # We concatenate all turns of the requested role to keep behavior robust.
    extracted_parts: list[str] = [
        str(turn.get("content", ""))
        for turn in conversation
        if turn.get("role") == role
    ]
    return "\n".join(extracted_parts).strip()


def _build_judge_user_prompt(
    problem_text: str,
    ground_truth: str,
    assistant_answer: str,
    void_answer: bool = False,
) -> str:
    if void_answer:
        return VOID_ANSWER_TOKEN

    # The assistant answer that the judge has access to should only be the part after the thinking.
    assistant_answer = (
        assistant_answer.split["</think>"][-1]
        if "</think>" in assistant_answer
        else assistant_answer
    )
    return (
        f"[Problem]\n{problem_text}\n\n"
        f"[Gold label Answer]\n{ground_truth}\n\n"
        f"[Model generated Answer]\n{assistant_answer}\n"
    )


def _extract_judge_text(response_json: dict[str, Any]) -> str:
    # OpenAI-compatible shape:
    #   response_json["choices"][0]["message"]["content"]
    choices = response_json.get("choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")

    # Some providers may return content as structured chunks instead of a string.
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _parse_score(judge_text: str) -> float:
    # We explicitly prompted the judge to put his answer in \boxed{}
    marker = r"\boxed{"
    idx = judge_text.rfind(marker)  # rightmost occurrence
    after = idx + len(marker)  # index of char immediately after "{"
    if idx != -1 and after < len(judge_text):
        if judge_text[after] in ["0", "1"]:
            return float(judge_text[after])

    # Fallback 1: Handle common "x/y" patterns (example: "7/10").
    fraction_match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", judge_text)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator != 0:
            return max(0.0, min(1.0, numerator / denominator))

    # Fallback 2: first scalar number in output.
    scalar_match = re.search(r"-?\d+(?:\.\d+)?", judge_text)
    if not scalar_match:
        return 0.0

    score = float(scalar_match.group(0))

    # If judge emits 0-10 scale (e.g. "8"), map it into 0-1.
    if 1.0 < score <= 10.0:
        score = score / 10.0

    return max(0.0, min(1.0, score))


@ray.remote(max_restarts=-1, max_task_retries=-1)
class LLMJudgeVerifyWorker:
    _REQUEST_TIMEOUT_SECONDS = 600

    def __init__(
        self,
        vllm_chat_completions_url: str,
        model: str,
        system_prompt: str,
        max_judge_output_tokens: int = 0,
    ) -> None:
        self.vllm_chat_completions_url = vllm_chat_completions_url
        self.model = model
        self.system_prompt = system_prompt
        self.max_judge_output_tokens = max_judge_output_tokens
        self._session = requests.Session()

    def _query_llm_judge(self, judge_prompt: str) -> tuple[float, str, str | None]:
        if VOID_ANSWER_TOKEN in judge_prompt:
            return (
                0.0,
                judge_prompt,
                "Answer was specifically instructed to be voided. Judge was not called.",
            )
        request_payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": judge_prompt},
            ],
            "model": self.model,
            "temperature": 0.0,
            "stream": False,
        }
        if self.max_judge_output_tokens > 0:
            request_payload["max_tokens"] = self.max_judge_output_tokens

        try:
            response = self._session.post(
                self.vllm_chat_completions_url,
                json=request_payload,
                timeout=self._REQUEST_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            response_json = response.json()
            judge_text = _extract_judge_text(response_json)
            score = _parse_score(judge_text)
            return score, judge_text, None
        except Exception as exc:
            return 0.0, "", str(exc)

    def verify(
        self, judge_prompt_batch: list[str]
    ) -> list[tuple[float, str, str | None]]:
        return [
            self._query_llm_judge(judge_prompt) for judge_prompt in judge_prompt_batch
        ]

    def shutdown(self) -> None:
        self._session.close()


@ray.remote(max_restarts=-1, max_task_retries=-1)
class LLMJudgeEnvironment(EnvironmentInterface[LLMJudgeMetadata]):
    """Single-turn reward environment that delegates scoring to LLM judge workers."""

    def __init__(self, cfg: LLMJudgeEnvConfig | dict[str, Any]):
        self.cfg = cfg

        self.vllm_server_url = str(cfg.get("vllm_server_url", "")).strip()
        self.model = str(cfg.get("model", "")).strip()
        self.system_prompt = str(
            cfg.get("system_prompt", DEFAULT_JUDGE_SYSTEM_PROMPT)
        ).strip()

        if not self.vllm_server_url:
            raise ValueError("llm_judge config requires `vllm_server_url`.")
        if not self.model:
            raise ValueError("llm_judge config requires non-empty `model`.")
        if not self.system_prompt:
            raise ValueError("llm_judge config requires non-empty `system_prompt`.")

        raw_num_workers = cfg.get("num_workers", 1)
        try:
            self.num_workers = int(raw_num_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"llm_judge config requires integer `num_workers`, got {raw_num_workers!r}."
            ) from exc

        if self.num_workers <= 0:
            raise ValueError(
                f"llm_judge config requires positive `num_workers`, got {self.num_workers}."
            )

        self.vllm_chat_completions_url = _normalize_to_chat_completions_url(
            self.vllm_server_url
        )
        self.workers = [
            LLMJudgeVerifyWorker.remote(
                self.vllm_chat_completions_url,
                self.model,
                self.system_prompt,
            )
            for _ in range(self.num_workers)
        ]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[LLMJudgeMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[LLMJudgeMetadata]:
        judge_prompt_batch: list[str] = []
        format_check_result_batch: list = []

        for conversation, env_info in zip(message_log_batch, metadata):
            # Pull prompt + model response from the conversation and target from metadata.
            problem_text = _extract_role_content(conversation, role="user")
            assistant_answer = _extract_role_content(conversation, role="assistant")
            # In normal usage, env_info is a dict with "ground_truth".
            # We still guard for malformed rows to avoid hard crashes.
            ground_truth = (
                str(env_info.get("ground_truth", ""))
                if isinstance(env_info, dict)
                else ""
            )
            format_check_result = verify_format(assistant_answer, "</think>")
            format_check_result_batch.append(format_check_result)

            judge_prompt = _build_judge_user_prompt(
                problem_text=problem_text,
                ground_truth=ground_truth,
                assistant_answer=assistant_answer,
                void_answer=bool(not format_check_result.is_correct_format),
            )
            judge_prompt_batch.append(judge_prompt)

        chunked_judge_prompt_batch = chunk_list_to_workers(
            judge_prompt_batch, self.num_workers
        )
        futures = [
            self.workers[i].verify.remote(chunk)
            for i, chunk in enumerate(chunked_judge_prompt_batch)
        ]
        worker_results = ray.get(futures)

        judge_results: list[tuple[float, str, str | None]] = []
        for worker_result in worker_results:
            judge_results.extend(worker_result)

        rewards = [score for score, _, _ in judge_results]

        def _build_observation(idx: int) -> dict[str, str]:
            score, judge_text, judge_error = judge_results[idx]
            format_check_result = format_check_result_batch[idx]

            if VOID_ANSWER_TOKEN not in judge_text:
                if judge_error is not None:
                    return {
                        "role": "environment",
                        "content": f"Environment 'llm_judge': llm_judge errored ({judge_error})\nNo formatting issues.",
                    }
                else:
                    # Keep feedback compact so logs remain readable.
                    judge_preview = judge_text.replace("\n", " ").strip()[:160]
                    return {
                        "role": "environment",
                        "content": (
                            f"Environment 'llm_judge': llm_judge_score={score:.3f}; "
                            f"judge_text='{judge_preview}'\nNo formatting issues."
                        ),
                    }
            else:
                issues = "\n".join(format_check_result.issues)
                observations.append(
                    {
                        "role": "environment",
                        "content": f"Environment 'llm_judge': incorrect format.\nFormatting issues:\n{issues}",
                    }
                )

        observations = [_build_observation(idx) for idx in range(judge_results)]

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminated_tensor = torch.ones_like(rewards_tensor).cpu()

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        # Mean judge score is the main metric to track for this reward.
        metrics = {
            "llm_judge_mean_reward": float(batch["rewards"].float().mean().item())
        }
        return batch, metrics

    def shutdown(self) -> None:
        # Shutdown all judge workers.
        ray.get([worker.shutdown.remote() for worker in self.workers])
        for worker in self.workers:
            ray.kill(worker)


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MockLLMJudgeEnvironment(EnvironmentInterface[LLMJudgeMetadata]):
    """Super-simple mock LLM judge.

    This class keeps the same init contract as `LLMJudgeEnvironment` but does
    not make any external requests. Every sample gets reward 0.67.
    """

    _MOCK_REWARD = 0.67

    def __init__(self, cfg: LLMJudgeEnvConfig | dict[str, Any]):
        # Keep the same config contract as the real judge env.
        self.cfg = cfg
        self.vllm_server_url = str(cfg.get("vllm_server_url", "")).strip()
        self.model = str(cfg.get("model", "")).strip()
        self.system_prompt = str(
            cfg.get("system_prompt", DEFAULT_JUDGE_SYSTEM_PROMPT)
        ).strip()

        if not self.vllm_server_url:
            raise ValueError("llm_judge config requires `vllm_server_url`.")
        if not self.model:
            raise ValueError("llm_judge config requires non-empty `model`.")
        if not self.system_prompt:
            raise ValueError("llm_judge config requires non-empty `system_prompt`.")

        # Keep the same URL normalization/validation behavior as real env.
        self.vllm_chat_completions_url = _normalize_to_chat_completions_url(
            self.vllm_server_url
        )

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[LLMJudgeMetadata],
    ) -> EnvironmentReturn[LLMJudgeMetadata]:
        # No external calls. Return a fixed reward for every sample.
        rewards_tensor = torch.full(
            (len(message_log_batch),),
            self._MOCK_REWARD,
            dtype=torch.float32,
        ).cpu()
        terminated_tensor = torch.ones_like(rewards_tensor).cpu()

        observations: list[dict[str, str]] = [
            {
                "role": "environment",
                "content": f"Environment: mock_llm_judge_score={self._MOCK_REWARD:.2f}",
            }
            for _ in message_log_batch
        ]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        metrics = {
            "llm_judge_mean_reward": float(batch["rewards"].float().mean().item()),
            "mock_llm_judge_constant_reward": self._MOCK_REWARD,
        }
        return batch, metrics

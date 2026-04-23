# Math + MCQ Multi-Task Example Data (Second Example)

This folder is intentionally separate from the first MCQ-only example.

## Files

- `math_train.jsonl`: training rows for math task
- `math_val.jsonl`: validation rows for math task
- `mcq_train.jsonl`: training rows for MCQ task
- `mcq_val.jsonl`: validation rows for MCQ task

## Row Format

Each row is one JSON object:

```json
{"input":"<prompt text>", "output":"<ground truth answer>"}
```

## Why This Works

- The config uses `dataset_name: ResponseDataset` + `input_key: input` + `output_key: output`.
- The processor `math_hf_data_processor` reads the row and places `output` into `extra_env_info.ground_truth`.
- The environment then compares model output against `ground_truth`.

## Task Routing

- Rows loaded from `math_*.jsonl` are routed to env `"math"`.
- Rows loaded from `mcq_*.jsonl` are routed to env `"mcq_exact"`.

Routing is configured in:

- `examples/configs/grpo_math_mcq_multitask.yaml`

and implemented by:

- `examples/run_grpo_math_mcq_multitask.py`

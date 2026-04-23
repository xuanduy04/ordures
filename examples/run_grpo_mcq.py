"""Entry point for running GRPO with the custom MCQ exact-match environment.

This file is intentionally tiny:
1) Register env name -> actor class path.
2) Reuse the project's standard examples.run_grpo main().
"""

from nemo_rl.environments.utils import register_env

# Register custom env name -> actor implementation before run_grpo imports config/env.
# Why here:
# - GRPO setup looks up envs by string name from config.
# - That lookup happens during run_grpo execution.
# - So registration must happen before main() starts loading/instantiating envs.
register_env(
    # This is the env name you must reference in YAML:
    #   data.default.env_name: "mcq_exact"
    env_name="mcq_exact",
    # Fully-qualified class name for dynamic import via Hydra utilities.
    # Format: "<python.module.path>.<ClassName>"
    actor_class_fqn="examples.mcq_exact_env.MCQExactMatchEnvironment",
)

# Importing main after registration keeps execution order obvious.
from examples.run_grpo import main


if __name__ == "__main__":
    # Delegates all argument parsing and training flow to existing run_grpo script.
    main()

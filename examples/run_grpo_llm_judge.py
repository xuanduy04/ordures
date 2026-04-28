from nemo_rl.environments.utils import register_env

register_env(
    env_name="llm_judge",
    actor_class_fqn="examples.custom_rewards.llm_judge_env.LLMJudgeEnvironment",
)

register_env(
    env_name="mcq_exact_math",
    actor_class_fqn="examples.custom_rewards.mcq_exact_math_env.MCQExactMatchEnvironment",
)

from examples.run_grpo import main


if __name__ == "__main__":
    main()

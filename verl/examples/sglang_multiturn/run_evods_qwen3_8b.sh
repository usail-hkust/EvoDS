# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='data_science' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.clean_tools=True \
    data.max_prompt_length=6144 \
    actor_rollout_ref.rollout.response_length=24576 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    reward_model.reward_manager=datascience \
    actor_rollout_ref.model.path="$PROJECT_DIR/../checkpoints/EvoDS-sft-8B" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=wandb \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='evods-rl' \
    trainer.experiment_name='evods-rl-qwen3-8b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_training_steps=300 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    data.train_files=data/rl/train_rl.parquet \
    data.val_files=data/rl/test_rl.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/dsagent_tool_config.yaml" \
    trainer.total_epochs=150 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 $@

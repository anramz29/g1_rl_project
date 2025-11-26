# g1_box_grasp_train.py

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from g1_box_grasp_env import G1BoxGraspEnv
import numpy as np
from datetime import datetime
import os

# If false, starts from scratch. If true, tries to start from the most recent checkpoint.
RESUME_FROM_CHECKPOINT = True  


# Define Run Name
timestamp = datetime.now().strftime("%d_%H_%M_%S")
n_steps = 2_000_000  # Start with 2M steps for initial testing
run_name = f"g1_box_grasp_{n_steps}_steps_{timestamp}"

# Log Directory
root = "rl_logs"
log_dir = os.path.join(root, run_name)

# Checkpoint Directory
checkpoint_dir = os.path.join("rl_models", "checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"Training Configuration:")
print(f"{'='*60}")
print(f"Task: Box Grasping and Lifting")
print(f"Total timesteps: {n_steps:,}")
print(f"Checkpoint frequency: every 20000 steps")
print(f"Checkpoint location: {checkpoint_dir}")
print(f"Expected checkpoints: {n_steps // 20000}")
print(f"Log directory: {log_dir}")
print(f"{'='*60}\n")

# Create environment (no rendering for speed)
env = G1BoxGraspEnv(render_mode=None) #replace this with "human" to see it train. alternatievly use g1_box_grasp_eval.py. Use None if you don't want to render.
obs, info = env.reset()


def find_latest_checkpoint_global():
    """Find the most recent checkpoint across ALL training runs"""
    checkpoint_base = "rl_models/checkpoints"
    if not os.path.exists(checkpoint_base):
        return None
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(checkpoint_base) 
                      if os.path.isdir(os.path.join(checkpoint_base, d))]
    
    if not checkpoint_dirs:
        return None
    
    # Find latest checkpoint across ALL directories
    all_checkpoints = []
    for dir_name in checkpoint_dirs:
        dir_path = os.path.join(checkpoint_base, dir_name)
        checkpoints = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
        for cp in checkpoints:
            all_checkpoints.append(os.path.join(dir_path, cp))
    
    if not all_checkpoints:
        return None
    
    # Return most recent by modification time
    return max(all_checkpoints, key=lambda x: os.path.getmtime(x))


def get_steps_from_checkpoint(filename):
    """Extract step number from checkpoint filename"""
    parts = os.path.basename(filename).split('_')
    for i, part in enumerate(parts):
        if 'steps.zip' in part:
            return int(parts[i-1].replace('steps.zip', ''))
    return 0


# tries to start training from last checkpoint in case it takes multiple sittings to train
if RESUME_FROM_CHECKPOINT:
    latest_checkpoint = find_latest_checkpoint_global()
    if latest_checkpoint:
        print(f"✓ Resuming from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        
        # Calculate remaining steps
        completed_steps = get_steps_from_checkpoint(latest_checkpoint)
        remaining_steps = max(0, n_steps - completed_steps)
        
        print(f"Completed: {completed_steps:,} steps")
        print(f"Remaining: {remaining_steps:,} steps\n")
        
        if remaining_steps <= 0:
            print("Training already complete for this step count!")
            print(f"Current checkpoint: {completed_steps:,} / {n_steps:,} steps")
            env.close()
            exit(0)
    else:
        print("⚠ No checkpoint found, starting fresh\n")
        remaining_steps = n_steps
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=log_dir
        )
else:
    print("Starting fresh training (RESUME_FROM_CHECKPOINT = False)\n")
    remaining_steps = n_steps
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir
    )

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=checkpoint_dir,
    name_prefix="g1_box_grasp_checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print("Starting training...")
print("Use 'tensorboard --logdir rl_logs/' to monitor progress\n")

# Train for remaining steps
model.learn(
    total_timesteps=remaining_steps,
    callback=checkpoint_callback,
    tb_log_name=run_name,
    reset_num_timesteps=(not RESUME_FROM_CHECKPOINT or latest_checkpoint is None)
)

# Save final model
final_save_path = os.path.join("rl_models", run_name + "_final")
model.save(final_save_path)

print(f"\n{'='*60}")
print(f"✓ Training Complete!")
print(f"✓ Final model saved to: {final_save_path}")
print(f"✓ Checkpoints saved in: {checkpoint_dir}")
print(f"{'='*60}\n")

env.close()
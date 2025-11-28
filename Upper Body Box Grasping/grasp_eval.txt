# g1_box_grasp_eval.py

from stable_baselines3 import PPO
from g1_box_grasp_env import G1BoxGraspEnv
import sys
import os

def find_latest_model():
    """Find the most recent final model or checkpoint across all runs"""
    latest_model = None
    latest_time = 0
    
    # First, look for final models
    if os.path.exists("rl_models"):
        final_models = [f for f in os.listdir("rl_models") if f.endswith("_final.zip")]
        for model_file in final_models:
            model_path = os.path.join("rl_models", model_file)
            mod_time = os.path.getmtime(model_path)
            if mod_time > latest_time:
                latest_time = mod_time
                latest_model = model_path
    
    # If found a final model, return it (prefer final over checkpoints)
    if latest_model:
        return latest_model
    
    # Otherwise, look for latest checkpoint across ALL directories
    checkpoint_base = "rl_models/checkpoints"
    if os.path.exists(checkpoint_base):
        checkpoint_dirs = [d for d in os.listdir(checkpoint_base) 
                          if os.path.isdir(os.path.join(checkpoint_base, d))]
        
        for dir_name in checkpoint_dirs:
            dir_path = os.path.join(checkpoint_base, dir_name)
            checkpoints = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
            
            for cp_file in checkpoints:
                cp_path = os.path.join(dir_path, cp_file)
                mod_time = os.path.getmtime(cp_path)
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_model = cp_path
    
    return latest_model

if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    print("No model path provided, searching for latest model...")
    model_path = find_latest_model()
    
    if model_path is None:
        print("Error: No trained models found!")
        print("Train a model first using: python g1_box_grasp_train.py")
        sys.exit(1)
    
    print(f"Found: {model_path}")

print(f"\n{'='*60}")
print(f"Loading model: {model_path}")
print(f"{'='*60}\n")

# Create environment WITH rendering
eval_env = G1BoxGraspEnv(render_mode="human")



# Load trained model
model = PPO.load(model_path)

# Evaluate continuously until user stops (Ctrl+C)
print("Running continuous evaluation. Press Ctrl+C to stop.\n")

episode = 0
try:
    while True:
        episode += 1
        obs, info = eval_env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            eval_env.render()
        
        print(f"  Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final box height: {info['box_height']:.3f}m")
        print(f"  Left contact: {info['left_contact']}")
        print(f"  Right contact: {info['right_contact']}")

except KeyboardInterrupt:
    print("\n\nEvaluation stopped by user.")

eval_env.close()
print("✓ Evaluation complete!")


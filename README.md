# G1 RL Project

Reinforcement learning for the **Unitree G1 humanoid robot** to perform a **box grasping task** using MuJoCo and Stable-Baselines3.

## Technical Report

A detailed write-up covering environment design, reward shaping, and results is available here:

[G1 Box Grasp – Technical Report (PDF)](EECE5552_Final_Project_Report.pdf)

## Demo

[![G1 Box Grasp Demo](https://img.youtube.com/vi/kKOSwQv5tlA/0.jpg)](https://youtu.be/kKOSwQv5tlA)

## Overview

The robot's upper body (arms and hands) learns to approach, grasp, and lift a cardboard box from the sides using both arms. The lower body is locked during training to focus learning on arm control.

- **Simulator:** MuJoCo
- **RL Algorithm:** PPO (Proximal Policy Optimization) via Stable-Baselines3
- **Robot:** Unitree G1 (29-DOF with hands)
- **Task:** Dual-arm box grasp and lift

## Project Structure

```
g1_rl_project/
├── unitree_g1/
│   └── g1_mocap_29dof_with_hands.xml      # Base G1 MuJoCo model
└── Upper_Body/
    ├── g1_box_grasp_env_both_arms_adrian.py # Main RL environment
    ├── g1_box_grasp_train_adrian.py         # Training script
    ├── g1_box_grasp_eval.py                 # Evaluation script
    ├── g1_two_boxes_custom_keyframes_friction.xml  # MuJoCo scene with box
    ├── arms_down_keyframe.xml               # Initial keyframe pose
    ├── mujoco_robot_useful_methods.py       # MuJoCo helper utilities
    ├── g1_reward_debug.py                   # Reward debugging callback
    ├── reward_debug_callback.py             # SB3 training callback
    ├── meshes/                              # STL mesh files for G1 robot
    └── rl_models/                           # Saved model checkpoints
```

## Getting Started

### Dependencies

```bash
pip install mujoco gymnasium stable-baselines3 numpy
```

### Training

```bash
cd Upper_Body
python g1_box_grasp_train_adrian.py
```

Checkpoints are saved to `rl_models/checkpoints/` and the final model to `rl_models/`.

### Evaluation

```bash
cd Upper_Body
python g1_box_grasp_eval.py
```

The eval script automatically finds the latest trained model and renders the robot in the MuJoCo viewer.

## Environment Details

- **Observation space:** Joint positions/velocities, hand fingertip positions, box position/orientation
- **Action space:** Upper body joint torques (both arms)
- **Reward:** Shaped reward for approaching the box, making contact with both hands, and lifting to a target height
- **Episode termination:** Box lifted to target height (success) or timeout

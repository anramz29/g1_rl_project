# view_g1_scene.py
import time
import mujoco
import mujoco.viewer
import numpy as np

from g1_box_grasp_env_both_arms_adrian import G1BoxGraspEnv


def build_arm_mirror_map(model):
    """
    Build a list of (left_qpos_idx, right_qpos_idx, sign) tuples.
    sign = +1 for same rotation, -1 if the axis is mirrored (e.g. roll).
    """
    pairs = [
        ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint",  1),
        ("left_shoulder_roll_joint",  "right_shoulder_roll_joint",  -1),  # roll is mirrored
        ("left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",   1),
        ("left_elbow_joint",          "right_elbow_joint",          1),
        ("left_wrist_yaw_joint",      "right_wrist_yaw_joint",      1),
        # you can add wrists / fingers too if you want:
        # ("left_wrist_roll_joint",     "right_wrist_roll_joint",    -1),
        # ("left_wrist_pitch_joint",    "right_wrist_pitch_joint",   1),
    ]

    mirror_indices = []

    for left_name, right_name, sign in pairs:
        l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left_name)
        r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, right_name)

        l_q = model.jnt_qposadr[l_id]
        r_q = model.jnt_qposadr[r_id]

        mirror_indices.append((l_q, r_q, sign))

    return mirror_indices


def apply_qpos_mirroring(data, mirror_indices):
    """
    Copy left-arm joint positions to right arm each frame.
    """
    for l_q, r_q, sign in mirror_indices:
        data.qpos[r_q] = sign * data.qpos[l_q]


def main():
    # Create env WITHOUT viewer logic
    env = G1BoxGraspEnv(render_mode=None)
    obs, info = env.reset()  # sets keyframe, table, box, etc.

    # Precompute which joints to mirror
    mirror_indices = build_arm_mirror_map(env.model)

    # Launch a plain MuJoCo viewer on that model/data
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        while True:
            # --- MIRRORING HAPPENS HERE ---
            # Whatever you do to the left-arm joints gets copied to right
            apply_qpos_mirroring(env.data, mirror_indices)

            # Step physics (optional for posing; you can also pause with spacebar)
            mujoco.mj_step(env.model, env.data)

            viewer.sync()
            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()

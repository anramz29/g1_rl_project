# manual_reward_debugger_v2.py
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# Use your helper methods for placement and keyframes
from mujoco_robot_useful_methods import set_body_position, load_keyframe

# ---------------------------
# Helper: find the joint index for a body with freejoint (same logic as your env)
# ---------------------------
def find_body_joint(model, body_id):
    for i in range(model.njnt):
        if model.body_jntadr[body_id] <= i < model.body_jntadr[body_id] + model.body_jntnum[body_id]:
            return i
    return None

# ---------------------------
# Contact helper functions (copied / adapted)
# ---------------------------
def check_contact(data, model, body_name, geom_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]
        if ((body1 == body_id and geom2 == geom_id) or
            (body2 == body_id and geom1 == geom_id)):
            return True
    return False

def get_contact_force(data, model, body_name, geom_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        if ((body1 == body_id and geom2 == geom_id) or
            (body2 == body_id and geom1 == geom_id)):

            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            return np.linalg.norm(force[:3])

    return 0.0


# ---------------------------
# Reward function copied (comments preserved, updated to use xPos)
# ---------------------------
def calculate_reward(model, data):
    """
    Multi-component reward for box grasping and lifting
    
    Components:
    1. Reaching: hands approach box sides
    2. Contact: touching box (individual + bilateral bonus)
    3. Grip force: appropriate force when in contact
    4. Lift height: box height progress (unconditional)
    5. Stability: box position/orientation stable
    6. Control cost: penalize excessive actions
    7. Alive bonus: small reward for staying alive
    """
    reward = 0.0

    # Adjusted weights for better balance
    w_reach = 8.0
    w_contact = 5.0
    w_lift = 10.0
    w_force = 2.0
    w_stability = 0.5
    w_control = 0.5
    w_alive = 0.1  # Much smaller to avoid domination
    
    # Get positions using xPos (world space)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box")
    box_pos = data.xpos[box_body_id]  # Use world space position instead of qpos
    
    # Keep joint info for velocity calculations (used in commented sections)
    box_joint_id = find_body_joint(model, box_body_id)
    box_qpos_start = model.jnt_qposadr[box_joint_id]
    box_qvel_start = model.jnt_dofadr[box_joint_id]

    left_thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
    left_index_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_0_link")
    left_middle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link")

    right_thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
    right_index_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
    right_middle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link")

    left_hand_pos = (data.xpos[left_thumb_id] + 
                    data.xpos[left_index_id] + 
                    data.xpos[left_middle_id]) / 3.0 # average each hand's three finger 0 link positions to create a location roughly in the palm of the robot

    right_hand_pos = (data.xpos[right_thumb_id] + 
                    data.xpos[right_index_id] + 
                    data.xpos[right_middle_id]) / 3.0
    
    # 1. REACHING REWARD: Hands move toward box sides
    box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
    box_half_depth = model.geom_size[box_geom_id][1]

    xNudge = 0
    target_left = box_pos + np.array([xNudge, box_half_depth, 0])
    target_right = box_pos + np.array([xNudge,  -box_half_depth, 0])

    left_dist = np.linalg.norm(left_hand_pos - target_left)
    right_dist = np.linalg.norm(right_hand_pos - target_right)

    print("left distance ",  left_dist)
    
    """
    # --- New logic: choose linear vs exponential based on hand Z ---
    box_z = box_pos[2]
    z_low  = box_z - 0.02    # allowable vertical region

    def reach_term(hand_pos, target_pos):
        hand_z = hand_pos[2]

        if z_low < hand_z:
            # Use linear reward for strong horizontal motivation
            return 10.0 - 5.0 * np.linalg.norm(hand_pos - target_pos)
        else:
            # Use exponential reward for gentle shaping outside Z range
            return 2*np.exp(-0.75 * np.linalg.norm(hand_pos - target_pos))

    left_reach = reach_term(left_hand_pos, target_left)
    right_reach = reach_term(right_hand_pos, target_right)
    """
    
    
    left_reach = 5.0*(np.exp(-5 * left_dist))
    right_reach = 5.0*(np.exp(-5 * right_dist))
    


    reaching_reward = left_reach + right_reach
    reward += w_reach * reaching_reward

    # 2. CONTACT REWARD: Only reward if hands are near the target positions
    left_hand_bodies = [
        "left_wrist_yaw_link",
        "left_hand_palm_link", 
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
        "left_hand_index_0_link",
        "left_hand_index_1_link"
    ]

    right_hand_bodies = [
        "right_wrist_yaw_link",
        "right_hand_palm_link", 
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link"
    ]

    left_touching = any(check_contact(data, model, body, "box_geom") for body in left_hand_bodies)
    right_touching = any(check_contact(data, model, body, "box_geom") for body in right_hand_bodies)

    # Calculate target positions
    xNudge = 0
    target_left = box_pos + np.array([xNudge, box_half_depth, 0])
    target_right = box_pos + np.array([xNudge, -box_half_depth, 0])

    left_dist = np.linalg.norm(left_hand_pos - target_left)
    right_dist = np.linalg.norm(right_hand_pos - target_right)

    # Only reward contact if hand is NEAR the correct position
    contact_threshold = 0.08

    if left_touching and left_dist < contact_threshold:
        left_contact_reward = 10.0  # Good contact!
        print("left good contact")   
    elif left_touching:
        print("left bad contact")   
        left_contact_reward = -2.0  # Bad contact (wrong position)
    else:
        left_contact_reward = 0.0

    if right_touching and right_dist < contact_threshold:
        right_contact_reward = 10.0
        print("right good contact")   
    elif right_touching:
        right_contact_reward = -2.0
        print("right bad contact")  
    else:
        right_contact_reward = 0.0

    reward += w_contact * (left_contact_reward + right_contact_reward)

    # Bilateral bonus (only if BOTH in correct positions)
    if (left_touching and right_touching and 
        left_dist < contact_threshold and right_dist < contact_threshold):
        bilateral_bonus = 20.0
        reward += w_contact * bilateral_bonus


    # PALM ORIENTATION REWARD: Y-axis of hands parallel to world Y-axis
    left_rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")].reshape(3, 3)
    right_rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")].reshape(3, 3)

    # Get Y-axis of each hand
    left_palm_y = left_rot[:, 1]   # Y-axis column
    right_palm_y = right_rot[:, 1]

    # World Y-axis
    world_y = np.array([0, 1, 0])

    # Dot product measures alignment (-1 to 1)
    # Take absolute value since ±Y is both fine (hands face opposite directions)
    left_alignment = abs(np.dot(left_palm_y, world_y))
    right_alignment = abs(np.dot(right_palm_y, world_y))

    # Reward when palm Y-axis aligns with world Y (either +Y or -Y)
    # Value ranges 0 to 1, where 1 = perfectly aligned
    palm_orientation_reward = 2.0 * (left_alignment + right_alignment)  # Max = 4.0
    reward += palm_orientation_reward
    
    """
    # 3. GRIP FORCE REWARD: Only when in contact
    if left_touching or right_touching:
        # Use max force instead of sum to avoid inflated values
        left_forces = [get_contact_force(data, model, body, "box_geom") for body in left_hand_bodies]
        right_forces = [get_contact_force(data, model, body, "box_geom") for body in right_hand_bodies]
        
        left_force = max(left_forces + [0])
        right_force = max(right_forces + [0])
        total_force = left_force + right_force
        
        if total_force > 0:
            if total_force < self.min_grip_force:
                # Too weak - encourage more force
                force_reward = 2.0 * (total_force / self.min_grip_force)
            elif total_force > self.max_safe_grip_force * 4:
                # Way too strong - penalize heavily
                 force_reward = -5.0
            else:
                # In reasonable range - reward being close to ideal
                ideal_total = self.ideal_grip_force * 2  # Both hands   
                force_error = abs(total_force - ideal_total) / ideal_total
                force_reward = 3.0 * np.exp(-2.0 * force_error)
            
            reward += w_force * force_reward

    # 4. LIFT HEIGHT REWARD: Only when grasping (now uses xPos for current height)
    totalHeightReward = 0
    if left_touching or right_touching:
        current_height = box_pos[2]  # Using xpos instead of qpos
        height_progress = current_height - self.initial_box_pos[2]
        
        # Calculate target progress (how far it should lift)
        target_progress = self.target_lift_height - self.initial_box_pos[2] 
        
        # Reward progress up to target, penalize going beyond
        if height_progress <= target_progress:
            # Moving toward target - reward progress
            progress_reward = 133 * max(0, height_progress)
        else:
            # Exceeded target - penalize the overshoot
            overshoot = height_progress - target_progress
            progress_reward = 133 * target_progress - 50 * overshoot  # Penalty for going too high
        
        # Bonus for being at target
        height_error = abs(current_height - self.target_lift_height)
        if height_error < 0.05:
            target_bonus = 20.0
        else:
            target_bonus = 0.0
        
        totalHeightReward = w_lift * (progress_reward + target_bonus)
        reward += totalHeightReward
    
        # 5. STABILITY REWARDS: Only when box is lifted (uses both xPos and qPos)
        if height_progress > 0.05:  # Only care about stability when lifted
            # Box velocity should be low (still needs qvel)
            box_vel = data.qvel[box_qvel_start:box_qvel_start+6]
            box_speed = np.linalg.norm(box_vel[:3])
            velocity_reward = np.exp(-2.0 * box_speed)  # Reward low velocity
            
            # Box should maintain XY position (now uses xPos)
            xy_drift = np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2])
            position_reward = np.exp(-5.0 * xy_drift)  # Reward staying in place
            
            # Box orientation should stay upright (still needs quat from qpos)
            box_quat = data.qpos[box_qpos_start+3:box_qpos_start+7]
            orientation_reward = 2.0 * abs(box_quat[0])  # quat[0] should be ~1
            
            stability_reward = velocity_reward + position_reward + orientation_reward
            reward += w_stability * stability_reward
    """
    # 6. CONTROL COST: Penalize large actions
    # In this manual script we do not drive actuators (user moves manually),
    # still use data.ctrl for a small control penalty as in original code.
    control_cost = -np.sum(np.square(data.ctrl))
    reward += w_control * control_cost
    
    
    # 7. ALIVE BONUS: Small reward for staying alive
    reward += w_alive

    return reward


# ---------------------------
# MAIN
# ---------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use same XML path as g1_with_box_test.py
    xml_path = "g1_two_boxes_custom_keyframes.xml"
    os.environ.setdefault("MUJOCO_GL", "glfw")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Load keyframe using your helper (try both keys that appear in repo)
    try:
        load_keyframe(model, data, "stand")
    except Exception:
        try:
            load_keyframe(model, data, "stand_thumbs_open")
        except Exception:
            # fallback: mj_resetDataKeyframe if available
            try:
                key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
                mujoco.mj_resetDataKeyframe(model, data, key_id)
            except Exception:
                pass

    # Use your set_body_position helper (copied placements from env)
    try:
        set_body_position(model, data, "table_box", x=0.7, y=0.0, z=0.3)
        set_body_position(model, data, "cardboard_box", x=0.42, y=0.0, z=0.76)
    except Exception as e:
        print("set_body_position failed or body names differ:", e)

    # Forward to update transforms/geometry
    mujoco.mj_forward(model, data)

    # Print a small header
    print("\nManual reward debugger (v2) - Using xPos for world space positions")
    print("XML:", xml_path)
    print("Use the viewer to move the robot manually. Reward is printed each sim step.\n")

    # -------------------------------------------------------
    # FIND BASE JOINT (same as original env)
    # -------------------------------------------------------
    base_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    base_qpos_addr = model.jnt_qposadr[base_joint_id]

    # Store initial base pose
    fixed_base_qpos = data.qpos[base_qpos_addr:base_qpos_addr+7].copy()

    # Launch viewer and loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():

            # freeze floating base qpos: [x, y, z, qw, qx, qy, qz]
            data.qpos[base_qpos_addr : base_qpos_addr+7] = fixed_base_qpos

            # zero root velocity (base is not allowed to move)
            data.qvel[0:6] = 0


            # -------------------------------
            # MIRROR RIGHT ARM FROM LEFT ARM
            # -------------------------------

            # Actuator indices (same as your environment)
            left_arm_actuators  = [15, 16, 17, 18, 21]
            right_arm_actuators = [29, 30, 31, 32, 35]

            # Signs for mirroring symmetry (same as env)
            arm_mirror_signs = np.array([1, -1, -1, 1, -1])

            # Read left-arm actuator commands from data.ctrl
            left_ctrl = data.ctrl[left_arm_actuators]

            # Compute mirrored commands
            right_ctrl = left_ctrl * arm_mirror_signs

            # Apply to right arm
            data.ctrl[right_arm_actuators] = right_ctrl


            # Step physics
            mujoco.mj_step(model, data)

            # Update any visualization spheres if present (match g1_with_box_test naming)
            try:
                box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box")
                box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
                box_size = model.geom_size[box_geom_id]
                box_half_depth = box_size[1]
                box_pos = data.xpos[box_body_id]  # Using xpos here too

                # compute targets (same formula as your test file)
                xNudge = 0
                target_left = box_pos + np.array([xNudge, box_half_depth, 0])
                target_right = box_pos + np.array([xNudge,  -box_half_depth, 0])

                # try to move mocaps if they exist (visualization)
                try:
                    left_target_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_target_viz")]
                    right_target_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_target_viz")]
                    left_hand_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_viz")]
                    right_hand_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_viz")]

                    # average finger positions (using xpos)
                    left_thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
                    left_index_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_0_link")
                    left_middle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link")
                    right_thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
                    right_index_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
                    right_middle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link")

                    left_hand_pos = (data.xpos[left_thumb_id] + data.xpos[left_index_id] + data.xpos[left_middle_id]) / 3.0
                    right_hand_pos = (data.xpos[right_thumb_id] + data.xpos[right_index_id] + data.xpos[right_middle_id]) / 3.0

                    data.mocap_pos[left_target_mocap_id] = target_left
                    data.mocap_pos[right_target_mocap_id] = target_right
                    data.mocap_pos[left_hand_mocap_id] = left_hand_pos
                    data.mocap_pos[right_hand_mocap_id] = right_hand_pos
                except Exception:
                    # mocaps or bodies not present — ignore visualization update
                    pass

            except Exception:
                # geometry or naming not present — ignore
                pass

            # Calculate reward using the copied function
            r = calculate_reward(model, data)

            # Print per-step (not cumulative)
            #print(f"Step {step:06d} | Reward = {r:.6f}")

            step += 1
            # Sync viewer
            viewer.sync()
            # sleep a tiny bit to avoid huge spam (you can reduce)
            time.sleep(0.005)


if __name__ == "__main__":
    main()
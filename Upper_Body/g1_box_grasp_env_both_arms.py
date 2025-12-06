# g1_box_grasp_env.py

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import os
import time
from mujoco_robot_useful_methods import set_body_position, load_keyframe


class G1BoxGraspEnv(gym.Env):
    """
    RL Environment for G1 Humanoid Box Grasping Task
    
    Task: Grasp box from sides and lift to target height
    - Lower body locked (waist down)
    - Fingers locked in grasp-ready position
    """
    
    def __init__(self, render_mode='human', render_fps=30, policy_freq=50):
        super().__init__()
        
        # Load model with box
        xml_path = "g1_two_boxes_custom_keyframes.xml"
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Physics parameters
        self.frame_skip = int((1 / self.model.opt.timestep) / policy_freq)
        
        # Get box dimensions and friction from XML
        box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        self.mu_s = self.model.geom_friction[box_geom_id, 0]
        box_size = self.model.geom_size[box_geom_id]
        
        # Box dimensions (size in XML is "0.14 0.15 0.16" = X Y Z half-widths)
        self.box_half_width = box_size[0]   # X dimension
        self.box_half_depth = box_size[1]   # Y dimension (the sides to grasp!)
        self.box_half_height = box_size[2]  # Z dimension

        self.left_hand_bodies = [
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

        self.right_hand_bodies = [
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

        # Get body IDs
        self.left_hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.right_hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box")

        # Find box qpos/qvel indices
        box_joint_id = self._find_body_joint(self.box_body_id)
        self.box_qpos_start = self.model.jnt_qposadr[box_joint_id]
        self.box_qvel_start = self.model.jnt_dofadr[box_joint_id]

        # Initialize environment state first. This locks the robot, sets its keyframe (starting pos), and sets up the two boxes
        self._setup_initial_state()

        # NOW set task parameters after everything is positioned
        self.g = 9.81  # m/s^2
        self.max_episode_steps = 250
        self.current_step = 0

        # Get box mass from XML
        self.box_mass = self.model.body_mass[self.box_body_id]

        # Get initial box height and set target
        initial_box_height = self.data.qpos[self.box_qpos_start + 2]  # Z coordinate
        self.target_lift_height = initial_box_height + 0.3  # Lift 0.3m from starting height
        # I'm (for now at least) setting the training to lift the box by around a foot. because of the starting values this ends up being 1 meter total in the air.
        
        # Calculate required grip force (from ROM)
        self.min_grip_force = (self.box_mass * self.g) / (2 * self.mu_s)
        self.ideal_grip_force = self.min_grip_force * 1.5  # 50% safety margin
        self.max_safe_grip_force = self.min_grip_force * 2.0  # Don't exceed 3x
        
        print(f"\n{'='*60}")
        print(f"G1 Box Grasping Environment")
        print(f"{'='*60}")
        print(f"Box mass: {self.box_mass} kg")
        print(f"Friction coeff (μ_s): {self.mu_s}")
        print(f"Min grip force per hand: {self.min_grip_force:.2f} N")
        print(f"Ideal grip force per hand: {self.ideal_grip_force:.2f} N")
        print(f"Max safe grip force per hand: {self.max_safe_grip_force:.2f} N")
        print(f"Target lift height: {self.target_lift_height:.2f} m")
        print(f"{'='*60}\n")
        
        # Define actuator groups
        self.locked_actuators = list(range(0, 15))  # Legs + Waist
        self.left_arm_actuators = [15, 16, 17, 18, 21]  # shoulder pitch/roll/yaw, elbow, wrist_yaw
        self.right_arm_actuators = [29, 30, 31, 32, 35]  # shoulder pitch/roll/yaw, elbow, wrist_yaw
        self.left_wrist_locked = [19, 20]  # Lock wrist roll/pitch only
        self.right_wrist_locked = [33, 34]  # Lock wrist roll/pitch only
        self.hand_actuators = list(range(22, 29)) + list(range(36, 43))
 
        
        # ACTION SPACE: Control both arms (10 DOF total)
        both_arm_actuators = self.left_arm_actuators + self.right_arm_actuators

        low = self.model.actuator_ctrlrange[both_arm_actuators, 0]
        high = self.model.actuator_ctrlrange[both_arm_actuators, 1]

        self.action_space = Box(
            low=low,
            high=high,
            dtype=np.float64
        )
        self.both_arm_actuators = both_arm_actuators
        print(f"Action Space: {self.action_space.shape} (left arm: shoulder + elbow)")
        
        # =================================================================
        # OBSERVATION SPACE: Arms + Box state
        # ~60 dimensions total
        # =================================================================
        obs_dim = self._calculate_obs_dim()
        obs_low, obs_high = self._get_obs_limits()
        self.observation_space = Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float64
        )
        print(f"Observation Space: {self.observation_space.shape}")
        print(f"  - Left arm qpos (4) + qvel (4)")
        print(f"  - Right arm qpos (4) + qvel (4)")
        print(f"  - Box pos (3) + quat (4) + vel (6)")
        print(f"  - Hand positions (6)")
        print(f"  - Box-to-hand vectors (6)")
        print(f"  - Target info (3)")
        print(f"{'='*60}\n")
        
        """
        # Calculate max action change based on velocity limit
        max_arm_vel = 1.0  # rad/s (your desired max speed)
        step_duration = self.model.opt.timestep * self.frame_skip  # seconds per RL step
        self.max_action_change = max_arm_vel * step_duration  # radians per step
        
        print(f"Max arm velocity: {max_arm_vel} rad/s")
        print(f"Step duration: {step_duration:.4f} s")
        print(f"Max action change per step: {self.max_action_change:.4f} rad")
        
        self.previous_action = np.zeros(len(self.left_arm_actuators))
        """

        # bookkeeping for "moving away" penalty
        self.min_left_hand_dist = 1000
        self.min_right_hand_dist = 1000
        # weight for moving-away penalty (positive number); will subtract reward when moving away
        self.w_move_away = 1000.0
        
        # Rendering
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps

    def _get_obs_limits(self):
        """Set observation space limits"""
        obs_low = []
        obs_high = []
        
        # Arm position limits (from actuator ranges)
        for _ in range(2):  # left and right arms
            obs_low.extend(self.model.actuator_ctrlrange[self.left_arm_actuators, 0])
            obs_high.extend(self.model.actuator_ctrlrange[self.left_arm_actuators, 1])
        
        # Arm velocity limits (keep slow!)
        max_arm_vel = 1.0  # rad/s - adjust this to control speed of robot's arms
        obs_low.extend([-max_arm_vel] * 10)  # 5 DOF x 2 arms
        obs_high.extend([max_arm_vel] * 10)
        
        # Box state (position, orientation, velocity) - unbounded
        obs_low.extend([-np.inf] * 13)  # 3 pos + 4 quat + 6 vel
        obs_high.extend([np.inf] * 13)
        
        # Hand positions - unbounded
        obs_low.extend([-np.inf] * 6)
        obs_high.extend([np.inf] * 6)
        
        # Box-to-hand vectors - unbounded
        obs_low.extend([-np.inf] * 6)
        obs_high.extend([np.inf] * 6)
        
        # Target info - unbounded
        obs_low.extend([-np.inf] * 3)
        obs_high.extend([np.inf] * 3)
        
        return np.array(obs_low), np.array(obs_high)
    
    def _find_body_joint(self, body_id):
        """Find the joint ID for a body with a freejoint"""
        for i in range(self.model.njnt):
            if self.model.body_jntadr[body_id] <= i < self.model.body_jntadr[body_id] + self.model.body_jntnum[body_id]:
                return i
        return None
    
    def _calculate_obs_dim(self):
        """Calculate total observation dimensions"""
        dim = 0
        dim += 5  # Left arm qpos
        dim += 5  # Left arm qvel
        dim += 5  # Right arm qpos
        dim += 5  # Right arm qvel
        dim += 3  # Box position
        dim += 4  # Box quaternion
        dim += 6  # Box velocity (linear + angular)
        dim += 3  # Left hand position
        dim += 3  # Right hand position
        dim += 3  # Box to left hand vector
        dim += 3  # Box to right hand vector
        dim += 3  # Target info (target height, current height, height error)
        return dim
    
    def _setup_initial_state(self):
        """Load keyframe and setup initial positions"""
        # Load arms_bent_fingers_open keyframe
        load_keyframe(self.model, self.data, "arms_out_ready_to_grab") #stand_thumbs_open
        
        # Position table and box
        set_body_position(self.model, self.data, "table_box", x=0.7, y=0.0, z=0.3)
        set_body_position(self.model, self.data, "cardboard_box", x=0.38, y=0.0, z=0.76)

        
        # Fix robot base
        base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self.base_qpos_addr = self.model.jnt_qposadr[base_joint_id]
        self.fixed_base_qpos = self.data.qpos[self.base_qpos_addr:self.base_qpos_addr+7].copy()
        
        # Store standing control for locked joints
        self.standing_ctrl = self.data.ctrl.copy()
        
        # Store initial box position
        self.initial_box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3].copy()
    
    def get_obs(self):
        """
        Get current observation
        Returns: concatenated array of all observation components
        """
        # Left arm state (now 5 DOF with wrist yaw)
        left_arm_qpos_indices = [22, 23, 24, 25, 28]  # shoulder + elbow + wrist_yaw
        left_arm_qvel_indices = [21, 22, 23, 24, 27]
        left_arm_qpos = self.data.qpos[left_arm_qpos_indices]
        left_arm_qvel = self.data.qvel[left_arm_qvel_indices]
        
        # Right arm state (now 5 DOF with wrist yaw)
        right_arm_qpos_indices = [36, 37, 38, 39, 42]
        right_arm_qvel_indices = [35, 36, 37, 38, 41]
        right_arm_qpos = self.data.qpos[right_arm_qpos_indices]
        right_arm_qvel = self.data.qvel[right_arm_qvel_indices]
        
        # Box state
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]
        box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]
        box_vel = self.data.qvel[self.box_qvel_start:self.box_qvel_start+6]
        
        # Hand positions (average of thumb, index, middle finger bases)
        left_thumb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
        left_index_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_0_link")
        left_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link")
        
        right_thumb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
        right_index_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
        right_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link")
        
        left_hand_pos = (self.data.xpos[left_thumb_id] + 
                        self.data.xpos[left_index_id] + 
                        self.data.xpos[left_middle_id]) / 3.0
        
        right_hand_pos = (self.data.xpos[right_thumb_id] + 
                        self.data.xpos[right_index_id] + 
                        self.data.xpos[right_middle_id]) / 3.0
        
        # Relative vectors
        box_to_left = left_hand_pos - box_pos
        box_to_right = right_hand_pos - box_pos
        
        # Target info
        current_height = box_pos[2]
        height_error = self.target_lift_height - current_height
        target_info = np.array([self.target_lift_height, current_height, height_error])
        
        obs = np.concatenate([
            left_arm_qpos,
            left_arm_qvel,
            right_arm_qpos,
            right_arm_qvel,
            box_pos,
            box_quat,
            box_vel,
            left_hand_pos,
            right_hand_pos,
            box_to_left,
            box_to_right,
            target_info
        ])
        
        return obs.astype(np.float64)
    
    def check_contact(self, body_name, geom_name):
        """Check if body is in contact with geom"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            body1_from_geom1 = self.model.geom_bodyid[geom1]
            body1_from_geom2 = self.model.geom_bodyid[geom2]
            
            if ((body1_from_geom1 == body_id and geom2 == geom_id) or
                (body1_from_geom2 == body_id and geom1 == geom_id)):
                return True
        return False
    
    def get_contact_force(self, body_name, geom_name):
        """Get contact force magnitude between body and geom"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            body1_from_geom1 = self.model.geom_bodyid[geom1]
            body1_from_geom2 = self.model.geom_bodyid[geom2]
            
            if ((body1_from_geom1 == body_id and geom2 == geom_id) or
                (body1_from_geom2 == body_id and geom1 == geom_id)):
                # Get contact force
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                return np.linalg.norm(force[:3])  # Normal force magnitude
        
        return 0.0
    


    # =================================================================
    # REWARD FUNCTION
    # =================================================================
    def calculate_reward(self, action):
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
        w_reach = 15.0
        w_contact = 5.0
        w_lift = 10.0
        w_force = 2.0
        w_stability = 0.5
        w_control = 0.1
        w_alive = 0.1  # Much smaller to avoid domination
        w_vel = 0.0 # penalization for movement. trying to reduce flapping.
        
        # Get positions
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]

        left_thumb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
        left_index_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_0_link")
        left_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link")

        right_thumb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
        right_index_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
        right_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link")

        left_hand_pos = (self.data.xpos[left_thumb_id] + 
                        self.data.xpos[left_index_id] + 
                        self.data.xpos[left_middle_id]) / 3.0 # average each hand's three finger 0 link positions to create a location roughly in the palm of the robot

        right_hand_pos = (self.data.xpos[right_thumb_id] + 
                        self.data.xpos[right_index_id] + 
                        self.data.xpos[right_middle_id]) / 3.0
        
        """
        # Velocity Penalty (trying to reduce flapping / oscillating / unecessary movement)
        #left_wrist_vel  = np.linalg.norm(self.data.xvelp[self.left_hand_body_id])
        #right_wrist_vel = np.linalg.norm(self.data.xvelp[self.right_hand_body_id])
        #velocity_penalty = w_vel * (left_wrist_vel + right_wrist_vel)
        reward -= w_vel * np.sum(np.square(self.data.qvel))
        """
        
        # 1. REACHING REWARD: Hands move toward box sides
        xNudge = 0
        target_left = box_pos + np.array([xNudge, self.box_half_depth, 0])
        target_right = box_pos + np.array([xNudge,  -self.box_half_depth, 0])

        left_dist = np.linalg.norm(left_hand_pos - target_left)
        right_dist = np.linalg.norm(right_hand_pos - target_right)

        
        def reach_term(hand_pos, target_pos):
            hand_z = hand_pos[2]
            box_z = box_pos[2]
            z_low = box_z - 0.02
            dist = np.linalg.norm(hand_pos - target_pos)
            
            if hand_z > z_low:
                # Above box - MUCH stronger linear gradient
                return 50.0 - 20.0 * dist  # Big bonus + steep slope
            else:
                # Below box - weak gradient (avoid table)
                return 2.0 * np.exp(-0.75 * dist)

        left_reach = reach_term(left_hand_pos, target_left)
        right_reach = reach_term(right_hand_pos, target_right)
        """
        
        left_reach = 5.0*(np.exp(-5 * left_dist))
        right_reach = 5.0*(np.exp(-5 * right_dist))
        

        reaching_reward = left_reach + right_reach
        reward += w_reach * reaching_reward
        
        # ---- moving-away / negative relative velocity penalty ----
        # compute current distances
        cur_left_dist  = np.linalg.norm(left_hand_pos - target_left)
        cur_right_dist = np.linalg.norm(right_hand_pos - target_right)

        # safe default if prev not set (e.g. first step)
        if self.min_left_hand_dist is not None and self.min_right_hand_dist is not None:

            penalty_left = max(0, cur_left_dist - self.min_left_hand_dist)
            penalty_right = max(0, cur_right_dist - self.min_right_hand_dist)


            # Clip penalties to avoid huge negative spikes
            max_penalty = 50.0
            penalty_left = np.clip(penalty_left, 0.0, max_penalty)
            penalty_right = np.clip(penalty_right, 0.0, max_penalty)
            reachAway =  (penalty_left + penalty_right)

            reward -= reachAway
        
        # Add this to encourage any movement toward box
        box_approach_bonus = 5.0 * max(0, 1.0 - min(left_dist, right_dist))
        reward += box_approach_bonus
        """
        # 2. CONTACT REWARD: Only reward if hands are near the target positions
        left_touching = any(self.check_contact(body, "box_geom") for body in self.left_hand_bodies)
        right_touching = any(self.check_contact(body, "box_geom") for body in self.right_hand_bodies)

        """
        left_dist = np.linalg.norm(left_hand_pos - target_left)
        right_dist = np.linalg.norm(right_hand_pos - target_right)

        # Only reward contact if hand is NEAR the correct position
        contact_threshold = 0.08  # Within 8cm of target

        # More forgiving contact reward
        if left_touching and left_dist < contact_threshold:
            left_contact_reward = 10.0
        elif left_touching and left_dist < 0.15:  # Still close
            left_contact_reward = 2.0  # Small positive reward
        elif left_touching:
            left_contact_reward = -2.0  # Much smaller penalty
        else:
            left_contact_reward = 0.0

        # More forgiving contact reward
        if right_touching and right_dist < contact_threshold:
            right_contact_reward = 10.0
        elif right_touching and right_dist < 0.15:  # Still close
            right_contact_reward = 2.0  # Small positive reward
        elif right_touching:
            right_contact_reward = -2.0  # Much smaller penalty
        else:
            right_contact_reward = 0.0

        reward += w_contact * (left_contact_reward + right_contact_reward)

        # Bilateral bonus (only if BOTH in correct positions)
        if (left_touching and right_touching and 
            left_dist < contact_threshold and right_dist < contact_threshold):
            bilateral_bonus = 20.0
            reward += w_contact * bilateral_bonus
        """
        """
        # PALM ORIENTATION REWARD: Y-axis of hands parallel to world Y-axis
        left_rot = self.data.xmat[self.left_hand_body_id].reshape(3, 3)
        right_rot = self.data.xmat[self.right_hand_body_id].reshape(3, 3)

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
        """
        # 3. GRIP FORCE REWARD: Only when in contact
        if left_touching or right_touching:
            # Use max force instead of sum to avoid inflated values
            left_forces = [self.get_contact_force(body, "box_geom") for body in self.left_hand_bodies]
            right_forces = [self.get_contact_force(body, "box_geom") for body in self.right_hand_bodies]
            
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
        """
        # 4. LIFT HEIGHT REWARD: Only when grasping
        totalHeightReward = 0
        if left_touching and right_touching:
            current_height = box_pos[2]
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
        
            # 5. STABILITY REWARDS: Only when box is lifted
            if height_progress > 0.05:  # Only care about stability when lifted
                # Box velocity should be low
                box_vel = self.data.qvel[self.box_qvel_start:self.box_qvel_start+6]
                box_speed = np.linalg.norm(box_vel[:3])
                velocity_reward = np.exp(-2.0 * box_speed)  # Reward low velocity
                
                # Box should maintain XY position
                xy_drift = np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2])
                position_reward = np.exp(-5.0 * xy_drift)  # Reward staying in place
                
                # Box orientation should stay upright
                box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]
                orientation_reward = 2.0 * abs(box_quat[0])  # quat[0] should be ~1
                
                stability_reward = velocity_reward + position_reward + orientation_reward
                reward += w_stability * stability_reward
    
        # 6. CONTROL COST: Penalize large actions
        left_arm_ctrl = self.data.ctrl[self.left_arm_actuators]
        control_cost = -np.sum(np.square(left_arm_ctrl))
        #control_cost = -np.sum(np.square(action))
        reward += w_control * control_cost

        
        
        # 7. ALIVE BONUS: Small reward for staying alive
        reward += w_alive

        """
        self.min_left_hand_dist = min(self.min_left_hand_dist, cur_left_dist)
        self.min_right_hand_dist = min(self.min_right_hand_dist, cur_right_dist)
        """

        """
        if self.current_step % 100 == 0:  # Print every 100 steps to avoid spam
            print(f"Step {self.current_step:4d} | "
                f"Reach: {w_reach * reaching_reward:6.1f} | "
                f"Contact: {w_contact * (left_contact_reward + right_contact_reward):6.1f} | "
                #f"Lift: {totalHeightReward:6.1f} | "
                f"Alive: {w_alive:6.1f} | "
                f"Dist: {left_dist:6.1f} | "
                f"Control: {w_control*control_cost:6.1f} | "
                f"Reach Away: {reachAway:6.1f} | "
                f"Total: {reward:8.1f}")
        """
        
        
        
        return reward
    
    def check_contact_any_robot_part(self, geom_name): # helper method for termination conditions. We want to punisht the robot touching the table.
        """Check if any robot body is touching specified geom"""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                # Found contact with table - check if it's a robot body (not the box)
                other_geom = contact.geom2 if contact.geom1 == geom_id else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]
                
                # Exclude box and table itself
                if other_body != self.box_body_id and other_body != 0:  # 0 is world
                    return True
        
        return False
    
    def check_arm_self_collision(self):
        """Check if left arm touches right arm"""
        # Check any left hand body against any right hand body
        for left_body in self.left_hand_bodies:
            for right_body in self.right_hand_bodies:
                if self.check_body_to_body_contact(left_body, right_body):
                    return True
        return False

    def check_body_to_body_contact(self, body1_name, body2_name):
        """Check if two bodies are in contact"""
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            
            if ((geom1_body == body1_id and geom2_body == body2_id) or
                (geom1_body == body2_id and geom2_body == body1_id)):
                return True
        
        return False

    def terminate(self):
        """Check termination conditions"""
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]
        
        # Box fell below starting height
        if box_pos[2] < self.initial_box_pos[2] - 0.2:
            return True
        
        # Box moved too far from starting XY position
        xy_dist = np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2])
        if xy_dist > 0.15:
            return True
        
        # Box rotated too much from initial orientation
        box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]
        box_rot = np.zeros(9)
        mujoco.mju_quat2Mat(box_rot, box_quat)
        box_rot_mat = box_rot.reshape(3, 3)
        
        world_axes = np.eye(3)
        max_tilt_degrees = 15
        min_alignment = np.cos(np.radians(max_tilt_degrees))
        
        for i in range(3):
            box_axis = box_rot_mat[:, i]
            world_axis = world_axes[:, i]
            alignment = abs(np.dot(box_axis, world_axis))
            
            if alignment < min_alignment:
                return True
        
        # Arms collided with each other (hands smacked together)
        if self.check_arm_self_collision():
            return True
        
        # Robot bumped table
        if self.check_contact_any_robot_part("table_geom"):
            return True
        
        # Robot fell
        torso_height = self.data.qpos[2]
        if torso_height < 0.5:
            return True
        
        # Max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset to initial keyframe
        self._setup_initial_state()

        box_x_offset = np.random.uniform(0, 0)
        #box_y_offset = np.random.uniform(0, 0.2)

        set_body_position(self.model, self.data, "cardboard_box", x=0.38 + box_x_offset, y=0.0, z=0.76)
        mujoco.mj_forward(self.model, self.data)
        self.initial_box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3].copy()

        
        # Add small noise to arm positions
        noise_scale = 0.01
        left_arm_qpos_indices = [22, 23, 24, 25]
        for idx in left_arm_qpos_indices:
            self.data.qpos[idx] += np.random.uniform(-noise_scale, noise_scale)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        obs = self.get_obs()
        info = {}

        
        return obs, info
    
    def step(self, action):
        """Execute one step in environment"""
        self.current_step += 1
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action to left arm
        self.data.ctrl[self.both_arm_actuators] = action

    
        # Lock lower body
        self.data.ctrl[self.locked_actuators] = self.standing_ctrl[self.locked_actuators]
        
        # Lock wrists at fixed angles (palms facing inward)
        self.data.ctrl[self.left_wrist_locked] = self.standing_ctrl[self.left_wrist_locked]
        self.data.ctrl[self.right_wrist_locked] = self.standing_ctrl[self.right_wrist_locked]
        
        # Freeze all fingers
        self.data.ctrl[self.hand_actuators] = self.standing_ctrl[self.hand_actuators]
        
        # Fix base position
        self.data.qpos[self.base_qpos_addr:self.base_qpos_addr+7] = self.fixed_base_qpos
        self.data.qvel[0:6] = 0  # Zero base velocity
        
        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self.get_obs()
        
        # Calculate reward
        reward = self.calculate_reward(action)
        
        # Check termination
        terminated = self.terminate()
        truncated = False

        # Get mocap IDs
        left_target_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_target_viz")]
        right_target_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_target_viz")]
        left_hand_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_viz")]
        right_hand_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_viz")]
        
        # Calculate positions (same as in reward function)
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]
        
        # Hand positions (averaged)
        left_thumb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
        left_index_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_0_link")
        left_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link")
        
        right_thumb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
        right_index_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
        right_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link")
        
        left_hand_pos = (self.data.xpos[left_thumb_id] + 
                        self.data.xpos[left_index_id] + 
                        self.data.xpos[left_middle_id]) / 3.0
        
        right_hand_pos = (self.data.xpos[right_thumb_id] + 
                        self.data.xpos[right_index_id] + 
                        self.data.xpos[right_middle_id]) / 3.0

        # UPDATE VISUALIZATION SPHERES (if rendering)
        if self.render_mode == "human":
            # Target positions
            #xNudge = 3*self.box_half_depth/4
            xNudge = 0
            target_left = box_pos + np.array([xNudge, self.box_half_depth, 0])
            target_right = box_pos + np.array([xNudge, -self.box_half_depth, 0])
            
            # Update sphere positions
            self.data.mocap_pos[left_target_mocap_id] = target_left
            self.data.mocap_pos[right_target_mocap_id] = target_right
            self.data.mocap_pos[left_hand_mocap_id] = left_hand_pos
            self.data.mocap_pos[right_hand_mocap_id] = right_hand_pos
        
        # Info
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]
        info = {
            'box_height': box_pos[2],
            'step': self.current_step,
            'left_contact': any(self.check_contact(body, "box_geom") for body in self.left_hand_bodies),
            'right_contact': any(self.check_contact(body, "box_geom") for body in self.right_hand_bodies),
        }
                
        # Render if needed
        if self.render_mode == "human":
            self.render()

        box_to_left = left_hand_pos - box_pos
        box_to_right = right_hand_pos - box_pos

        left_dist = np.linalg.norm(box_to_left)
        right_dist = np.linalg.norm(box_to_right)
        
        self.min_left_hand_dist = min(self.min_left_hand_dist, left_dist)
        self.min_right_hand_dist = min(self.min_right_hand_dist, right_dist)
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            time.sleep(self.render_dt)
            self.viewer.sync()
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            if self.render_mode == "human":
                self.viewer.close()
            self.viewer = None
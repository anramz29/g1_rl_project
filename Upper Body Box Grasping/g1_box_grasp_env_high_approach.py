# g1_box_grasp_env_high_approach.py

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
    - Arms encouraged to approach OVER the table, then down
    """

    def __init__(self, render_mode='human', render_fps=30, policy_freq=50):
        super().__init__()

        # Load model with box
        xml_path = "g1_two_boxes_custom_keyframes.xml"
        os.environ.setdefault("MUJOCO_GL", "glfw")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ---- NEW: table top height and hand safety plane ------------------
        table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        # Approximate table top: geom center z + half-height in z
        self.table_top_z = (self.model.geom_pos[table_geom_id, 2] +
                            self.model.geom_size[table_geom_id, 2])
        # Hands should stay above this plane (slightly above table)
        self.hand_safety_z = self.table_top_z + 0.02   # 2 cm above table
        # -------------------------------------------------------------------

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

        # ---- NEW: Table height and safety margin ---------------------------------
        table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        # z position of table geom center + its half-height in z => approximate top surface
        self.table_top_z = self.model.geom_pos[table_geom_id, 2] + self.model.geom_size[table_geom_id, 2]
        self.table_safety_margin = 0.03  # 3 cm above table to discourage hands going below
        # Pre-grasp vertical offset above box center where hands should initially approach
        self.pre_grasp_offset_z = 0.10   # 10 cm above box center
        # ---------------------------------------------------------------------------

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

        # Subset: finger bodies only (no wrist or palm)
        self.left_finger_bodies = [
            "left_hand_thumb_0_link",
            "left_hand_thumb_1_link",
            "left_hand_thumb_2_link",
            "left_hand_middle_0_link",
            "left_hand_middle_1_link",
            "left_hand_index_0_link",
            "left_hand_index_1_link",
        ]

        self.right_finger_bodies = [
            "right_hand_thumb_0_link",
            "right_hand_thumb_1_link",
            "right_hand_thumb_2_link",
            "right_hand_middle_0_link",
            "right_hand_middle_1_link",
            "right_hand_index_0_link",
            "right_hand_index_1_link",
        ]

        # Get body IDs
        self.left_hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.right_hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box")

        # Find box qpos/qvel indices
        box_joint_id = self._find_body_joint(self.box_body_id)
        self.box_qpos_start = self.model.jnt_qposadr[box_joint_id]
        self.box_qvel_start = self.model.jnt_dofadr[box_joint_id]

        # Initialize environment state first. This locks the robot, sets its keyframe, sets table/box
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

        # Calculate required grip force (from ROM)
        self.min_grip_force = (self.box_mass * self.g) / (2 * self.mu_s)
        self.ideal_grip_force = self.min_grip_force * 1.5  # 50% safety margin
        self.max_safe_grip_force = self.min_grip_force * 2.0  # Don't exceed 3x

        print(f"\n{'='*60}")
        print(f"G1 Box Grasping Environment")
        print(f"{'='*60}")
        print(f"Box mass: {self.box_mass} kg")
        print(f"Friction coeff (μ_s): {self.mu_s}")
        print(f"Table top z: {self.table_top_z:.3f} m  (safety margin: {self.table_safety_margin:.3f} m)")
        print(f"Pre-grasp offset above box: {self.pre_grasp_offset_z:.3f} m")
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
        print(f"Action Space: {self.action_space.shape} (both arms)")

        # =================================================================
        # OBSERVATION SPACE: Arms + Box state
        # =================================================================
        obs_dim = self._calculate_obs_dim()
        obs_low, obs_high = self._get_obs_limits()
        self.observation_space = Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float64
        )
        print(f"Observation Space: {self.observation_space.shape}")
        print(f"{'='*60}\n")

        # bookkeeping for "moving away" penalty (currently unused)
        self.min_left_hand_dist = 1000
        self.min_right_hand_dist = 1000
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
        max_arm_vel = 1.0  # rad/s
        obs_low.extend([-max_arm_vel] * 10)  # 5 DOF x 2 arms
        obs_high.extend([max_arm_vel] * 10)

        # Arm torque/force observation (unbounded or large bounds)
        obs_low.extend([-np.inf] * 10)
        obs_high.extend([np.inf] * 10)

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
            if (self.model.body_jntadr[body_id] <= i <
                    self.model.body_jntadr[body_id] + self.model.body_jntnum[body_id]):
                return i
        return None

    def _calculate_obs_dim(self):
        """Calculate total observation dimensions"""
        dim = 0
        dim += 5  # Left arm qpos
        dim += 5  # Left arm qvel
        dim += 5  # Right arm qpos
        dim += 5  # Right arm qvel
        dim += 10  # Arm actuator torques (5 DOF x 2 arms)
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
        # Start with standing pose, arms down, thumbs open
        load_keyframe(self.model, self.data, "stand_thumbs_open")

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

        # Arm actuator torques/forces for both arms
        arm_torques = self.data.actuator_force[self.both_arm_actuators]

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
            arm_torques,
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
    
    def _get_hand_positions(self):
        """Return (left_hand_pos, right_hand_pos) in world coordinates."""
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

        return left_hand_pos, right_hand_pos


    # =================================================================
    # REWARD FUNCTION
    # =================================================================
    def calculate_reward(self, action):
        """
        Multi-component reward for box grasping and lifting

        Components:
        1. Reaching: hands approach box sides from ABOVE table
        2. Contact: more fingers touching box
        3. Lift height: box height progress (once clearly above table)
        4. Stability: box stable while lifted
        5. Control cost: penalize excessive actions
        6. Alive bonus: small reward per step
        """
        reward = 0.0

        # Weights
        w_reach = 15.0
        w_grip = 5.0
        w_contact = 5.0
        w_lift = 10.0
        w_force = 2.0
        w_stability = 0.5
        w_control = 0.1
        w_alive = 0.1
        w_vel = 0.0
        w_fingers = 3.0
        w_hold = 0.5   # <-- give this a nonzero value to encourage "stop and hold"

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
                         self.data.xpos[left_middle_id]) / 3.0

        right_hand_pos = (self.data.xpos[right_thumb_id] +
                          self.data.xpos[right_index_id] +
                          self.data.xpos[right_middle_id]) / 3.0

        # -----------------------------------------------------------
        # HAND VELOCITY ESTIMATE (for speed penalty near box)
        # -----------------------------------------------------------
        if not hasattr(self, "_prev_left_hand_pos_reward"):
            self._prev_left_hand_pos_reward = left_hand_pos.copy()
            self._prev_right_hand_pos_reward = right_hand_pos.copy()

        dt = self.model.opt.timestep * self.frame_skip

        left_hand_vel = (left_hand_pos - self._prev_left_hand_pos_reward) / dt
        right_hand_vel = (right_hand_pos - self._prev_right_hand_pos_reward) / dt

        self._prev_left_hand_pos_reward = left_hand_pos.copy()
        self._prev_right_hand_pos_reward = right_hand_pos.copy()

        # -----------------------------------------------------------
        # HARD SAFETY PLANE PENALTY: hands going too low near table
        # -----------------------------------------------------------
        min_hand_z = min(left_hand_pos[2], right_hand_pos[2])
        if min_hand_z < self.hand_safety_z:
            penetration = self.hand_safety_z - min_hand_z
            reward -= 200.0 * penetration

        # -------------------------------
        # 1. REACHING REWARD (HIGH APPROACH)
        # -------------------------------
        xNudge = 0.0
        target_left = box_pos + np.array([xNudge,  self.box_half_depth,  self.pre_grasp_offset_z])
        target_right = box_pos + np.array([xNudge, -self.box_half_depth, self.pre_grasp_offset_z])

        # penalize high speed when close to box
        dist_left = np.linalg.norm(left_hand_pos - box_pos)
        dist_right = np.linalg.norm(right_hand_pos - box_pos)
        near_thresh = 0.15  # 15cm around box
        speed_penalty = 0.0

        if dist_left < near_thresh:
            speed_penalty += np.linalg.norm(left_hand_vel)
        if dist_right < near_thresh:
            speed_penalty += np.linalg.norm(right_hand_vel)

        w_speed = 0.5
        reward -= w_speed * speed_penalty

        def reach_term(hand_pos, target_pos):
            hand_z = hand_pos[2]
            box_z = box_pos[2]
            dist = np.linalg.norm(hand_pos - target_pos)

            base = 40.0 * np.exp(-4.0 * dist)

            dz = hand_z - box_z
            height_band = max(self.box_half_height, 0.05)
            dz_norm = dz / height_band

            vertical_factor = np.exp(-(dz_norm ** 2))
            return base * vertical_factor

        left_reach = reach_term(left_hand_pos, target_left)
        right_reach = reach_term(right_hand_pos, target_right)
        reaching_reward = left_reach + right_reach   # we'll apply this after we know grasp_on

        # -------------------------------
        # 2. CONTACT / FINGER REWARD
        # -------------------------------
        left_touching = any(self.check_contact(body, "box_geom") for body in self.left_hand_bodies)
        right_touching = any(self.check_contact(body, "box_geom") for body in self.right_hand_bodies)

        finger_contact_count = 0
        for body in (self.left_finger_bodies + self.right_finger_bodies):
            if self.check_contact(body, "box_geom"):
                finger_contact_count += 1

        max_fingers = len(self.left_finger_bodies) + len(self.right_finger_bodies)
        fraction_fingers = (finger_contact_count / max_fingers) if max_fingers > 0 else 0.0

        finger_contact_reward = fraction_fingers
        reward += w_fingers * finger_contact_reward

        # -----------------------------------------------------------
        # GOOD vs BAD CONTACT: side grasps vs top/front whacks
        # -----------------------------------------------------------
        box_half_depth = self.box_half_depth
        box_half_height = self.box_half_height

        bad_contact_count = 0
        good_contact_count = 0

        def classify_hand_contact(hand_pos, touching):
            if not touching:
                return 0, 0

            rel = hand_pos - box_pos
            side_dist_y = abs(rel[1])
            height_rel = rel[2]

            is_side = side_dist_y > 0.6 * box_half_depth
            is_mid_height = abs(height_rel) < 0.5 * box_half_height

            hit_from_top = height_rel > 0.7 * box_half_height
            front_back = side_dist_y < 0.3 * box_half_depth

            good = 1 if (is_side and is_mid_height and not hit_from_top) else 0
            bad = 1 if (hit_from_top or (front_back and not is_side)) else 0

            return good, bad

        g_l, b_l = classify_hand_contact(left_hand_pos, left_touching)
        g_r, b_r = classify_hand_contact(right_hand_pos, right_touching)
        good_contact_count = g_l + g_r
        bad_contact_count = b_l + b_r

        w_good_contact = 8.0
        w_bad_contact = 25.0

        reward += w_good_contact * good_contact_count
        reward -= w_bad_contact * bad_contact_count

        # --- GRASP PHASE LOGIC ---
        grasp_on = (good_contact_count == 2)

        # Once grasped, reduce importance of reaching targets
        reach_scale = 0.3 if grasp_on else 1.0
        reward += w_reach * reach_scale * reaching_reward

        # Once grasped, penalize hand motion so it "stops" and squeezes
        if grasp_on:
            hold_penalty = np.linalg.norm(left_hand_vel) + np.linalg.norm(right_hand_vel)
            reward -= w_hold * hold_penalty

        # -------------------------------
        # 3. GRIP + LIFT HEIGHT REWARD
        # Only when both hands are touching
        # -------------------------------
        totalHeightReward = 0.0
        if left_touching and right_touching:
            # --- NEW: grip force reward ---
            # Sum normal forces from finger bodies on each side
            left_force = sum(
                self.get_contact_force(body, "box_geom")
                for body in self.left_finger_bodies
            )
            right_force = sum(
                self.get_contact_force(body, "box_geom")
                for body in self.right_finger_bodies
            )

            def grip_score(F):
                """
                Score for a single hand:
                - Penalty if force < min_grip_force  (too weak)
                - Penalty if force > max_safe_grip_force (too strong)
                - Max reward near ideal_grip_force
                """
                # too weak
                if F < self.min_grip_force:
                    return - (self.min_grip_force - F) / max(self.min_grip_force, 1e-6)
                # too strong
                if F > self.max_safe_grip_force:
                    return - (F - self.max_safe_grip_force) / max(self.max_safe_grip_force, 1e-6)
                # in [min, max_safe]: peak at ideal
                return 1.0 - abs(F - self.ideal_grip_force) / max(self.ideal_grip_force, 1e-6)

            grip_reward = grip_score(left_force) + grip_score(right_force)
            reward += w_grip * grip_reward
            # --- END NEW GRIP REWARD ---

            # --- EXISTING HEIGHT REWARD LOGIC ---
            current_height = box_pos[2]
            height_progress = current_height - self.initial_box_pos[2]
            target_progress = self.target_lift_height - self.initial_box_pos[2]

            # Only reward lifting once box is above table + small margin
            if current_height > self.table_top_z + 0.02:
                if height_progress <= target_progress:
                    progress_reward = 133 * max(0, height_progress)
                else:
                    overshoot = height_progress - target_progress
                    progress_reward = 133 * target_progress - 50 * overshoot

                height_error = abs(current_height - self.target_lift_height)
                target_bonus = 20.0 if height_error < 0.05 else 0.0

                totalHeightReward = w_lift * (progress_reward + target_bonus)
                reward += totalHeightReward
            else:
                # If it's trying to drag while still essentially on table, penalize a bit
                reward -= 5.0

            # 4. STABILITY REWARD (when lifted)
            if height_progress > 0.05:
                box_vel = self.data.qvel[self.box_qvel_start:self.box_qvel_start+6]
                box_speed = np.linalg.norm(box_vel[:3])
                velocity_reward = np.exp(-2.0 * box_speed)

                xy_drift = np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2])
                position_reward = np.exp(-5.0 * xy_drift)

                box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]
                orientation_reward = 2.0 * abs(box_quat[0])  # upright

                stability_reward = velocity_reward + position_reward + orientation_reward
                reward += w_stability * stability_reward

        # 5. CONTROL COST: Penalize large actions
        left_arm_ctrl = self.data.ctrl[self.left_arm_actuators]
        control_cost = -np.sum(np.square(left_arm_ctrl))
        reward += w_control * control_cost

        # 6. ALIVE BONUS
        reward += w_alive

        return reward

    def check_contact_any_robot_part(self, geom_name):
        """Check if any robot body is touching specified geom"""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                other_geom = contact.geom2 if contact.geom1 == geom_id else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]

                if other_body != self.box_body_id and other_body != 0:  # 0 is world
                    return True

        return False

    def check_arm_self_collision(self):
        """Check if left arm touches right arm"""
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

        # -----------------------------------------------------------
        # EARLY TERMINATION: hands have dipped into/under the table
        # -----------------------------------------------------------
        left_hand_pos, right_hand_pos = self._get_hand_positions()
        min_hand_z = min(left_hand_pos[2], right_hand_pos[2])

        # Give a few steps of grace at the very start if you want
        if self.current_step > 5 and min_hand_z < self.table_top_z:
            return True


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

        # Arms collided with each other
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

        self._setup_initial_state()

        box_x_offset = np.random.uniform(0, 0)
        set_body_position(self.model, self.data, "cardboard_box", x=0.38 + box_x_offset, y=0.0, z=0.76)
        mujoco.mj_forward(self.model, self.data)
        self.initial_box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3].copy()

        # Add small noise to arm positions (left arm indices)
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

        ## NEW ##
        ACTION_SCALE = 0.3  # try 0.2–0.3 to start

        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply action to arms
        self.data.ctrl[self.both_arm_actuators] = action * ACTION_SCALE 

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

        # Get mocap IDs for viz
        left_target_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_target_viz")
        ]
        right_target_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_target_viz")
        ]
        left_hand_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_viz")
        ]
        right_hand_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_viz")
        ]

        # Positions
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]

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
            xNudge = 0.0
            target_left = box_pos + np.array([xNudge,  self.box_half_depth,  self.pre_grasp_offset_z])
            target_right = box_pos + np.array([xNudge, -self.box_half_depth, self.pre_grasp_offset_z])

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

        # Update min distances bookkeeping (currently unused in reward)
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

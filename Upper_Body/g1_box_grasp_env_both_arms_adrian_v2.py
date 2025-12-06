# g1_box_grasp_env.py

import os
import time
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer

from mujoco_robot_useful_methods import set_body_position, load_keyframe


class G1BoxGraspEnv(gym.Env):
    """
    RL Environment for G1 Humanoid Box Grasping Task.

    Task:
        - Grasp box from the sides with both hands.
        - Lift to a target height.
        - Lower body is locked (waist down).
        - Fingers are locked in a grasp-ready position.
    """

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(self, render_mode: str = None, render_fps: int = 30, policy_freq: int = 50):
        super().__init__()

        # -----------------------------
        # Mujoco model / physics setup
        # -----------------------------
        xml_path = "g1_two_boxes_custom_keyframes.xml"
        os.environ.setdefault("MUJOCO_GL", "glfw")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # How many sim steps per policy step?
        self.frame_skip = int((1.0 / self.model.opt.timestep) / policy_freq)

        # Cache IDs, dimensions, etc.
        self._init_body_and_geom_handles()

        # Initialize environment state (keyframe, table/box positions, base, etc.)
        self._setup_initial_state()

        # -----------------------------
        # Task parameters
        # -----------------------------
        self.g = 9.81  # gravity
        self.max_episode_steps = 250
        self.current_step = 0

        # Box mass and target lift height
        self.box_mass = self.model.body_mass[self.box_body_id]
        initial_box_height = self.data.qpos[self.box_qpos_start + 2]  # z coordinate
        self.target_lift_height = initial_box_height + 0.3  # lift 0.3m above start

        # Grip force requirements (ROM-based)
        self.min_grip_force = (self.box_mass * self.g) / (2 * self.mu_s)
        self.ideal_grip_force = self.min_grip_force * 1.5     # 50% safety margin
        self.max_safe_grip_force = self.min_grip_force * 2.0  # don't exceed 3x

        self._print_task_summary()

        # -----------------------------
        # Actuators and action space
        # -----------------------------
        self._init_actuator_groups()
        self._init_action_space()

        # -----------------------------
        # Observation space
        # -----------------------------
        obs_dim = self._calculate_obs_dim()
        obs_low, obs_high = self._get_obs_limits()
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float64)

        print(f"Observation Space: {self.observation_space.shape}")
        print("  - Left arm qpos (5) + qvel (5)")
        print("  - Right arm qpos (5) + qvel (5)")
        print("  - Arm actuator torques (10)")
        print("  - Box pos (3) + quat (4) + vel (6)")
        print("  - Hand positions (6)")
        print("  - Box-to-hand vectors (6)")
        print("  - Target info (3)")
        print("=" * 60 + "\n")

        # -----------------------------
        # Reward hyperparameters
        # -----------------------------
        self.w_reach = 2.0
        self.w_contact = 5.0   # reserved
        self.w_lift = 12.0
        self.w_force = 2.0     # reserved
        self.w_stability = 0.5
        self.w_control = 0.2
        self.w_alive = 0.1
        self.w_vel = 1e-3
        self.w_fingers = 6.0
        
        # action scaling
        self.action_scale = 0.02

        # Sliding penalty
        self.slide_margin = 0.03
        self.w_slide = 5.0
        self.far_slide_threshold = 0.08
        self.slide_margin = 0.01          # start penalizing after ~1 cm
        self.w_slide = 10.0               # harsher shaping for drift
        self.far_slide_threshold = 0.05 

        # Success criteria
        self.success_lift = 0.12      # m above initial
        self.max_xy_success = 0.05    # m drift
        self.max_tilt_deg_success = 20.0

        # Debug printing for reward components
        self.debug_rewards = False

        # -----------------------------
        # Misc bookkeeping
        # -----------------------------
        self.min_left_hand_dist = 1000.0
        self.min_right_hand_dist = 1000.0
        self.w_move_away = 1000.0  # unused currently, but kept for compatibility

        # Rendering related
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps

    # ------------------------------------------------------------------
    # INITIALIZATION HELPERS
    # ------------------------------------------------------------------
    def _init_body_and_geom_handles(self):
        """Cache IDs and geometry-related constants used throughout."""

        # Box geom/body and friction
        box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box")

        self.mu_s = self.model.geom_friction[box_geom_id, 0]
        box_size = self.model.geom_size[box_geom_id]  # half-widths [x, y, z]

        self.box_half_width = box_size[0]   # X dimension
        self.box_half_depth = box_size[1]   # Y dimension (sides to grasp)
        self.box_half_height = box_size[2]  # Z dimension

        # Box joint indices; assumes free joint
        box_joint_id = self._find_body_joint(self.box_body_id)
        self.box_qpos_start = self.model.jnt_qposadr[box_joint_id]
        self.box_qvel_start = self.model.jnt_dofadr[box_joint_id]

        # Hand body names
        self.left_hand_bodies = [
            "left_wrist_yaw_link",
            "left_hand_palm_link",
            "left_hand_thumb_0_link",
            "left_hand_thumb_1_link",
            "left_hand_thumb_2_link",
            "left_hand_middle_0_link",
            "left_hand_middle_1_link",
            "left_hand_index_0_link",
            "left_hand_index_1_link",
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
            "right_hand_index_1_link",
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

        # Body IDs for wrists (used for velocities, etc.)
        self.left_hand_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        self.right_hand_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )

        # Geom IDs
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

    def _print_task_summary(self):
        print("\n" + "=" * 60)
        print("G1 Box Grasping Environment")
        print("=" * 60)
        print(f"Box mass: {self.box_mass:.3f} kg")
        print(f"Friction coeff (μ_s): {self.mu_s:.3f}")
        print(f"Min grip force per hand: {self.min_grip_force:.2f} N")
        print(f"Ideal grip force per hand: {self.ideal_grip_force:.2f} N")
        print(f"Max safe grip force per hand: {self.max_safe_grip_force:.2f} N")
        print(f"Target lift height: {self.target_lift_height:.2f} m")
        print("=" * 60 + "\n")

    def _init_actuator_groups(self):
        """Define actuator group indices."""
        # Legs + waist (locked)
        self.locked_actuators = list(range(0, 15))

        # Arms (shoulder pitch/roll/yaw, elbow, wrist_yaw)
        self.left_arm_actuators = [15, 16, 17, 18, 21]
        self.right_arm_actuators = [29, 30, 31, 32, 35]
        self.both_arm_actuators = self.left_arm_actuators + self.right_arm_actuators

        # Lock wrist roll/pitch only
        self.left_wrist_locked = [19, 20]
        self.right_wrist_locked = [33, 34]

        # Hand actuators (fingers)
        self.hand_actuators = list(range(22, 29)) + list(range(36, 43))

    def _init_action_space(self):
        low = self.model.actuator_ctrlrange[self.both_arm_actuators, 0]
        high = self.model.actuator_ctrlrange[self.both_arm_actuators, 1]
        self.action_space = Box(low=low, high=high, dtype=np.float64)
        print(f"Action Space: {self.action_space.shape} (both arms: shoulder + elbow + wrist_yaw)")

    def _setup_initial_state(self):
        """Load keyframe and setup initial positions."""
        load_keyframe(self.model, self.data, "stand_thumbs_open")

        # Position table and box
        set_body_position(self.model, self.data, "table_box", x=0.7, y=0.0, z=0.3)
        set_body_position(self.model, self.data, "cardboard_box", x=0.38, y=0.0, z=0.76)

        # Fix robot base (floating base joint)
        base_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        self.base_qpos_addr = self.model.jnt_qposadr[base_joint_id]
        self.fixed_base_qpos = self.data.qpos[self.base_qpos_addr : self.base_qpos_addr + 7].copy()

        # Store standing control for locked joints
        self.standing_ctrl = self.data.ctrl.copy()

        # Store initial box position
        self.initial_box_pos = self.data.qpos[self.box_qpos_start : self.box_qpos_start + 3].copy()

    # ------------------------------------------------------------------
    # BASIC UTILITIES / STATE ACCESSORS
    # ------------------------------------------------------------------
    def _get_obs_limits(self):
        obs_low = []
        obs_high = []

        # Arm joint positions (use left arm range for both)
        for _ in range(2):  # left + right
            obs_low.extend(self.model.actuator_ctrlrange[self.left_arm_actuators, 0])
            obs_high.extend(self.model.actuator_ctrlrange[self.left_arm_actuators, 1])

        # Arm velocities
        max_arm_vel = 1.0  # rad/s
        obs_low.extend([-max_arm_vel] * 10)
        obs_high.extend([max_arm_vel] * 10)

        # Arm torques
        obs_low.extend([-np.inf] * 10)
        obs_high.extend([np.inf] * 10)

        # Box state (pos, quat, vel)
        obs_low.extend([-np.inf] * 13)
        obs_high.extend([np.inf] * 13)

        # Hand positions
        obs_low.extend([-np.inf] * 6)
        obs_high.extend([np.inf] * 6)

        # Box-to-hand vectors
        obs_low.extend([-np.inf] * 6)
        obs_high.extend([np.inf] * 6)

        # Target info
        obs_low.extend([-np.inf] * 3)
        obs_high.extend([np.inf] * 3)

        return np.array(obs_low), np.array(obs_high)

    def _find_body_joint(self, body_id: int):
        for i in range(self.model.njnt):
            start = self.model.body_jntadr[body_id]
            num = self.model.body_jntnum[body_id]
            if start <= i < start + num:
                return i
        return None

    def _calculate_obs_dim(self) -> int:
        dim = 0
        dim += 5  # left qpos
        dim += 5  # left qvel
        dim += 5  # right qpos
        dim += 5  # right qvel
        dim += 10  # torques
        dim += 3   # box pos
        dim += 4   # box quat
        dim += 6   # box vel
        dim += 3   # left hand pos
        dim += 3   # right hand pos
        dim += 3   # box->left
        dim += 3   # box->right
        dim += 3   # target info
        return dim

    def _get_arm_state(self):
        left_arm_qpos_indices = [22, 23, 24, 25, 28]
        left_arm_qvel_indices = [21, 22, 23, 24, 27]
        right_arm_qpos_indices = [36, 37, 38, 39, 42]
        right_arm_qvel_indices = [35, 36, 37, 38, 41]

        left_arm_qpos = self.data.qpos[left_arm_qpos_indices]
        left_arm_qvel = self.data.qvel[left_arm_qvel_indices]
        right_arm_qpos = self.data.qpos[right_arm_qpos_indices]
        right_arm_qvel = self.data.qvel[right_arm_qvel_indices]
        return left_arm_qpos, left_arm_qvel, right_arm_qpos, right_arm_qvel

    def _get_arm_torques(self):
        return self.data.actuator_force[self.both_arm_actuators]

    def _get_box_state(self):
        box_pos = self.data.qpos[self.box_qpos_start : self.box_qpos_start + 3]
        box_quat = self.data.qpos[self.box_qpos_start + 3 : self.box_qpos_start + 7]
        box_vel = self.data.qvel[self.box_qvel_start : self.box_qvel_start + 6]
        return box_pos, box_quat, box_vel

    def _get_hand_positions(self):
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

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------
    def get_obs(self) -> np.ndarray:
        left_arm_qpos, left_arm_qvel, right_arm_qpos, right_arm_qvel = self._get_arm_state()
        arm_torques = self._get_arm_torques()
        box_pos, box_quat, box_vel = self._get_box_state()
        left_hand_pos, right_hand_pos = self._get_hand_positions()

        box_to_left = left_hand_pos - box_pos
        box_to_right = right_hand_pos - box_pos

        current_height = box_pos[2]
        height_error = self.target_lift_height - current_height
        target_info = np.array([self.target_lift_height, current_height, height_error])

        obs = np.concatenate(
            [
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
                target_info,
            ]
        )
        return obs.astype(np.float64)

    # ------------------------------------------------------------------
    # CONTACT HELPERS
    # ------------------------------------------------------------------
    def check_contact(self, body_name: str, geom_name: str) -> bool:
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

    def get_contact_force(self, body_name: str, geom_name: str) -> float:
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
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                return np.linalg.norm(force[:3])
        return 0.0

    def check_contact_any_robot_part(self, geom_name: str) -> bool:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                other_geom = contact.geom2 if contact.geom1 == geom_id else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]
                if other_body != self.box_body_id and other_body != 0:
                    return True
        return False

    def check_body_to_body_contact(self, body1_name: str, body2_name: str) -> bool:
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

    def check_arm_self_collision(self) -> bool:
        for left_body in self.left_hand_bodies:
            for right_body in self.right_hand_bodies:
                if self.check_body_to_body_contact(left_body, right_body):
                    return True
        return False

    # ------------------------------------------------------------------
    # REWARD COMPONENT HELPERS
    # ------------------------------------------------------------------
    def _reward_penalties(self, box_pos) -> tuple[float, float, float]:
        """
        Sliding, table collision, velocity penalties.
        Returns (penalty, xy_drift, excess_drift).
        """
        reward = 0.0

        # Horizontal drift of box
        xy_drift = np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2])
        excess_drift = max(0.0, xy_drift - self.slide_margin)

        reward -= self.w_slide * excess_drift
        if xy_drift > self.far_slide_threshold:
            reward -= 50.0

        # Table collision
        if self.check_contact_any_robot_part("table_geom"):
            reward -= 20.0

        # Velocity penalty (joint flapping)
        reward -= self.w_vel * np.sum(np.square(self.data.qvel))

        return reward, xy_drift, excess_drift

    def _reward_reach(self, box_pos, left_hand_pos, right_hand_pos) -> float:
        xNudge = 0.0
        target_left = box_pos + np.array([xNudge, self.box_half_depth, 0.0])
        target_right = box_pos + np.array([xNudge, -self.box_half_depth, 0.0])

        def reach_term(hand_pos, target_pos):
            hand_z = hand_pos[2]
            box_z = box_pos[2]
            safe_z = box_z + 0.1

            # Phase 1: lift up to safe height
            if hand_z < safe_z:
                low_z = box_z - 0.20
                clamped = np.clip(hand_z, low_z, safe_z)
                progress = (clamped - low_z) / max(1e-3, (safe_z - low_z))
                return 5.0 * progress

            # Phase 2: focus on XY alignment over box side
            delta_xy = (hand_pos - target_pos)[:2]
            dist_xy = np.linalg.norm(delta_xy)
            return 10.0 * np.exp(-5.0 * dist_xy)

        reach_left = reach_term(left_hand_pos, target_left)
        reach_right = reach_term(right_hand_pos, target_right)
        return reach_left + reach_right

    def _reward_contacts(self) -> tuple[bool, bool, float]:
        """
        Returns (left_touching, right_touching, finger_fraction)
        """
        left_touching = any(self.check_contact(body, "box_geom") for body in self.left_hand_bodies)
        right_touching = any(self.check_contact(body, "box_geom") for body in self.right_hand_bodies)

        finger_contact_count = 0
        for body in self.left_finger_bodies + self.right_finger_bodies:
            if self.check_contact(body, "box_geom"):
                finger_contact_count += 1

        max_fingers = len(self.left_finger_bodies) + len(self.right_finger_bodies)
        finger_fraction = finger_contact_count / max_fingers if max_fingers > 0 else 0.0
        return left_touching, right_touching, finger_fraction

    def _reward_lift_and_stability(
        self,
        box_pos,
        box_quat,
        box_vel,
        xy_drift: float,
        left_touching: bool,
        right_touching: bool,
    ) -> tuple[float, float, bool]:
        """
        Returns (height_reward_contrib, stability_reward_contrib, success_flag),
        where height/stability are already scaled by w_lift / w_stability.
        """
        total_height_reward = 0.0
        stability_reward_scaled = 0.0
        success = False

        if not (left_touching and right_touching):
            return total_height_reward, stability_reward_scaled, success

        current_height = box_pos[2]
        height_progress = current_height - self.initial_box_pos[2]
        target_progress = self.target_lift_height - self.initial_box_pos[2]

        if height_progress <= target_progress:
            progress_reward = 133.0 * max(0.0, height_progress)
        else:
            overshoot = height_progress - target_progress
            progress_reward = 133.0 * target_progress - 50.0 * overshoot

        height_error = abs(current_height - self.target_lift_height)
        target_bonus = 20.0 if height_error < 0.05 else 0.0

        total_height_reward = self.w_lift * (progress_reward + target_bonus)

        # Stability once lifted
        if height_progress > 0.05:
            box_speed = np.linalg.norm(box_vel[:3])
            velocity_reward = np.exp(-2.0 * box_speed)
            position_reward = np.exp(-5.0 * xy_drift)
            orientation_reward = 2.0 * abs(box_quat[0])

            stability_raw = velocity_reward + position_reward + orientation_reward
            stability_reward_scaled = self.w_stability * stability_raw

            # Success check
            tilt_angle = 2.0 * np.arccos(np.clip(box_quat[0], -1.0, 1.0))
            success = (
                height_progress > self.success_lift
                and tilt_angle < np.deg2rad(self.max_tilt_deg_success)
                and xy_drift < self.max_xy_success
            )

        return total_height_reward, stability_reward_scaled, success

    def _reward_control_and_alive(self) -> float:
        both_ctrl = np.concatenate(
            [self.data.ctrl[self.left_arm_actuators], self.data.ctrl[self.right_arm_actuators]]
        )
        control_cost = -np.sum(np.square(both_ctrl))
        return self.w_control * control_cost + self.w_alive

    # ------------------------------------------------------------------
    # REWARD FUNCTION (high-level)
    # ------------------------------------------------------------------
    def calculate_reward(self, action: np.ndarray) -> float:
        """
        Multi-component reward for box grasping and lifting.
        """
        box_pos, box_quat, box_vel = self._get_box_state()
        left_hand_pos, right_hand_pos = self._get_hand_positions()

        # Penalties & basic kinematics
        reward, xy_drift, excess_drift = self._reward_penalties(box_pos)

        # Reaching
        reach_raw = self._reward_reach(box_pos, left_hand_pos, right_hand_pos)
        reward += self.w_reach * reach_raw

        # Contacts
        left_touching, right_touching, finger_fraction = self._reward_contacts()
        reward += self.w_fingers * finger_fraction

        # Lift + stability + success bonus
        height_term, stability_term, success = self._reward_lift_and_stability(
            box_pos, box_quat, box_vel, xy_drift, left_touching, right_touching
        )
        reward += height_term + stability_term

        if success and not getattr(self, "already_succeeded", False):
            reward += 1000.0
            self.already_succeeded = True

        # Control regularization + alive bonus
        reward += self._reward_control_and_alive()

        # Optional debug prints
        if self.debug_rewards and (self.current_step % 20 == 0):
            print(
                f"[step {self.current_step}] "
                f"reach={self.w_reach * reach_raw:.1f}, "
                f"fingers={self.w_fingers * finger_fraction:.1f}, "
                f"height={height_term:.1f}, "
                f"stab={stability_term:.1f}, "
                f"slide_penalty={-self.w_slide * excess_drift:.1f}, "
                f"success={self.already_succeeded}"
            )

        return reward

    # ------------------------------------------------------------------
    # TERMINATION LOGIC
    # ------------------------------------------------------------------
    def terminate(self) -> bool:
        """Legacy helper, delegates to _compute_termination for compatibility."""
        return self._compute_termination()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        self.already_succeeded = False
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self._setup_initial_state()

        box_x_offset = np.random.uniform(0, 0)
        set_body_position(
            self.model, self.data, "cardboard_box",
            x=0.38 + box_x_offset, y=0.0, z=0.76
        )

        mujoco.mj_forward(self.model, self.data)
        self.initial_box_pos = self.data.qpos[self.box_qpos_start : self.box_qpos_start + 3].copy()

        noise_scale = 0.01
        left_arm_qpos_indices = [22, 23, 24, 25]
        for idx in left_arm_qpos_indices:
            self.data.qpos[idx] += np.random.uniform(-noise_scale, noise_scale)

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        obs = self.get_obs()
        info = {}

        self.min_left_hand_dist = 1000.0
        self.min_right_hand_dist = 1000.0

        return obs, info

    def _compute_termination(self) -> bool:
        # success ends the episode too
        if getattr(self, "already_succeeded", False):
            return True

        box_pos, box_quat, _ = self._get_box_state()

        # 1) Box dropped too low
        if box_pos[2] < self.initial_box_pos[2] - 0.05:
            return True

        # 2) Box slid too far in XY
        if np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2]) > 0.025:  # ~1 inch
            return True

        # Box tilted too much
        tilt_angle = 2 * np.arccos(np.clip(box_quat[0], -1.0, 1.0))
        if tilt_angle > np.deg2rad(45.0):
            return True

        # Robot hits table
        if (self.current_step > 150_000 and \
            self.check_contact_any_robot_part("table_geom")
            ):
            return True

        # Arm self-collision or torso fall can be added here if you want

        return False

    # ------------------------------------------------------------------
    # STEP / RENDER / CLOSE
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """Execute one environment step given an action."""
        self.current_step += 1

        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply arm controls
        self.data.ctrl[self.both_arm_actuators] = action * self.action_scale

        # Lock lower body, wrists, and fingers to standing control
        self.data.ctrl[self.locked_actuators] = self.standing_ctrl[self.locked_actuators]
        self.data.ctrl[self.left_wrist_locked] = self.standing_ctrl[self.left_wrist_locked]
        self.data.ctrl[self.right_wrist_locked] = self.standing_ctrl[self.right_wrist_locked]
        self.data.ctrl[self.hand_actuators] = self.standing_ctrl[self.hand_actuators]

        # Fix base pose and zero base velocity
        self.data.qpos[self.base_qpos_addr : self.base_qpos_addr + 7] = self.fixed_base_qpos
        self.data.qvel[0:6] = 0.0

        # Step MuJoCo physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Observation and reward
        obs = self.get_obs()
        reward = self.calculate_reward(action)

        # Termination
        terminated = self._compute_termination()
        truncated = self.current_step >= self.max_episode_steps

        # Positions for visualization and info
        box_pos, _, _ = self._get_box_state()
        left_hand_pos, right_hand_pos = self._get_hand_positions()

        if self.render_mode == "human":
            self._update_mocap_visualization(box_pos, left_hand_pos, right_hand_pos)

        info = {
            "box_height": box_pos[2],
            "step": self.current_step,
            "left_contact": any(self.check_contact(body, "box_geom") for body in self.left_hand_bodies),
            "right_contact": any(self.check_contact(body, "box_geom") for body in self.right_hand_bodies),
        }

        if self.render_mode == "human":
            self.render()

        # Track best hand distances (for diagnostics)
        box_to_left = left_hand_pos - box_pos
        box_to_right = right_hand_pos - box_pos
        left_dist = np.linalg.norm(box_to_left)
        right_dist = np.linalg.norm(box_to_right)

        self.min_left_hand_dist = min(self.min_left_hand_dist, left_dist)
        self.min_right_hand_dist = min(self.min_right_hand_dist, right_dist)

        return obs, reward, terminated, truncated, info

    def _update_mocap_visualization(self, box_pos, left_hand_pos, right_hand_pos):
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

        xNudge = 0.0
        target_left = box_pos + np.array([xNudge, self.box_half_depth, 0.0])
        target_right = box_pos + np.array([xNudge, -self.box_half_depth, 0.0])

        self.data.mocap_pos[left_target_mocap_id] = target_left
        self.data.mocap_pos[right_target_mocap_id] = target_right
        self.data.mocap_pos[left_hand_mocap_id] = left_hand_pos
        self.data.mocap_pos[right_hand_mocap_id] = right_hand_pos

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            time.sleep(self.render_dt)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None and self.render_mode == "human":
            self.viewer.close()
        self.viewer = None

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
        xml_path = "g1_two_boxes_custom_keyframes_friction.xml"
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # -----------------------------
        # Geom / body ids for contacts & positions
        # -----------------------------
        # Box
        self.box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom"
        )
        self.box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box"
        )

        # Left hand thumb + index fingertip bodies (for positions)
        self.left_thumb_tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_finger_tip"
        )
        self.left_index_tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_finger_tip"
        )

        # If you want right hand too, add similar ids here
        self.right_thumb_tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_finger_tip"
        )
        self.right_index_tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_finger_tip"
        )


        # Geoms that represent thumb/finger contact surfaces
        # (if you added fingertip spheres with class="finger_collision", name them!)
        # For now, use the whole thumb links as collision if you don't have separate geoms.
        self.left_thumb_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_thumb_0_link"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_thumb_1_link"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_thumb_2_link"),
        ]
        self.left_index_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_index_0_link"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_index_1_link"),
        ]


        # -----------------------------
        # Wrist joint indices (qpos / qvel)
        # -----------------------------
        # Left wrist joints
        left_wrist_roll_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_roll_joint"
        )
        left_wrist_pitch_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_pitch_joint"
        )
        left_wrist_yaw_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_yaw_joint"
        )

        self.left_wrist_roll_qpos_idx = self.model.jnt_qposadr[left_wrist_roll_joint_id]
        self.left_wrist_pitch_qpos_idx = self.model.jnt_qposadr[left_wrist_pitch_joint_id]
        self.left_wrist_yaw_qpos_idx = self.model.jnt_qposadr[left_wrist_yaw_joint_id]

        self.left_wrist_roll_qvel_idx = self.model.jnt_dofadr[left_wrist_roll_joint_id]
        self.left_wrist_pitch_qvel_idx = self.model.jnt_dofadr[left_wrist_pitch_joint_id]
        self.left_wrist_yaw_qvel_idx = self.model.jnt_dofadr[left_wrist_yaw_joint_id]

        # Right wrist joints
        right_wrist_roll_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_roll_joint"
        )
        right_wrist_pitch_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_pitch_joint"
        )
        right_wrist_yaw_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_yaw_joint"
        )

        self.right_wrist_roll_qpos_idx = self.model.jnt_qposadr[right_wrist_roll_joint_id]
        self.right_wrist_pitch_qpos_idx = self.model.jnt_qposadr[right_wrist_pitch_joint_id]
        self.right_wrist_yaw_qpos_idx = self.model.jnt_qposadr[right_wrist_yaw_joint_id]

        self.right_wrist_roll_qvel_idx = self.model.jnt_dofadr[right_wrist_roll_joint_id]
        self.right_wrist_pitch_qvel_idx = self.model.jnt_dofadr[right_wrist_pitch_joint_id]
        self.right_wrist_yaw_qvel_idx = self.model.jnt_dofadr[right_wrist_yaw_joint_id]

        # -----------------------------
        # Shoulder joint indices (qpos)
        # -----------------------------
        # NOTE: replace JOINT_NAME_HERE with your actual joint names
        left_shoulder_pitch_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint"
        )
        left_shoulder_roll_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint"
        )

        right_shoulder_pitch_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint"
        )
        right_shoulder_roll_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint"
        )

        self.left_shoulder_pitch_qpos_idx  = self.model.jnt_qposadr[left_shoulder_pitch_joint_id]
        self.left_shoulder_roll_qpos_idx   = self.model.jnt_qposadr[left_shoulder_roll_joint_id]
        self.right_shoulder_pitch_qpos_idx = self.model.jnt_qposadr[right_shoulder_pitch_joint_id]
        self.right_shoulder_roll_qpos_idx  = self.model.jnt_qposadr[right_shoulder_roll_joint_id]


        
        # Physics parameters
        self.frame_skip = int((1 / self.model.opt.timestep) / policy_freq)
        
        # Get box dimensions and friction from XML
        box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        self.mu_s = self.model.geom_friction[box_geom_id, 0]
        box_size = self.model.geom_size[box_geom_id]
        
        # Box dimensions (size in XML is "0.14 0.15 0.16" = X Y Z half-widths)
        self.box_half_width = box_size[0]   # X dimension
        self.box_half_depth = box_size[1]
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

        # Contact tracking
        self.had_bilateral_contact = False

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
        self.target_lift_height = initial_box_height + 0.5  # Lift 0.3m from starting height
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
        self.left_arm_actuators = [15, 16, 17, 18]  # shoulder pitch/roll/yaw, elbow, wrist_yaw
        self.right_arm_actuators = [29, 30, 31, 32]  # shoulder pitch/roll/yaw, elbow, wrist_yaw
        self.left_wrist_locked = [19, 20]  # Lock wrist roll/pitch only
        self.right_wrist_locked = [33, 34]  # Lock wrist roll/pitch only
        self.hand_actuators = list(range(22, 29)) + list(range(36, 43))
 
        
        # ACTION SPACE: Control both arms (10 DOF total)
        self.controlled_actuators = (
            self.left_arm_actuators
             + self.right_arm_actuators
        )

        low = self.model.actuator_ctrlrange[self.controlled_actuators, 0]
        high = self.model.actuator_ctrlrange[self.controlled_actuators, 1]

        self.action_space = Box(low=low,high=high,dtype=np.float64)
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
        self.w_move_away = 2.0

        # Debug: store last reward breakdown
        self.last_reward_debug = {}

        # termination reason
        self.term_reason = ""

        
        # Rendering
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps

    def _get_obs_limits(self):
        """Set observation space limits to match get_obs() layout."""
        obs_low = []
        obs_high = []

        # -----------------------------
        # 1) Left arm qpos (5)
        # -----------------------------
        # Simple symmetric joint limits; you can tighten later or read from model.
        max_qpos = 2.0   # radians
        obs_low.extend([-max_qpos] * 4)
        obs_high.extend([ max_qpos] * 4)

        # -----------------------------
        # 2) Left arm qvel (5)
        # -----------------------------
        max_arm_vel = 1.0  # rad/s
        obs_low.extend([-max_arm_vel] * 4)
        obs_high.extend([ max_arm_vel] * 4)

        # -----------------------------
        # 3) Right arm qpos (5)
        # -----------------------------
        obs_low.extend([-max_qpos] * 4)
        obs_high.extend([ max_qpos] * 4)

        # -----------------------------
        # 4) Right arm qvel (5)
        # -----------------------------
        obs_low.extend([-max_arm_vel] * 4)
        obs_high.extend([ max_arm_vel] * 4)

        # -----------------------------
        # 5) Arm + hand torques/forces (len(self.controlled_actuators))
        # -----------------------------
        # You can replace 5.0 with something derived from actuator_forcerange.
        max_torque = 25.0
        torque_dim = len(self.controlled_actuators)
        obs_low.extend([-max_torque] * torque_dim)
        obs_high.extend([ max_torque] * torque_dim)

        # -----------------------------
        # 6) Box position (3), quaternion (4), velocity (6)
        # -----------------------------
        # Position: moderately bounded workspace
        max_pos = 2.0
        obs_low.extend([-max_pos] * 3)
        obs_high.extend([ max_pos] * 3)

        # Quaternion components in [-1, 1]
        obs_low.extend([-1.0] * 4)
        obs_high.extend([ 1.0] * 4)

        # Box linear + angular velocity (6) – leave fairly wide
        max_box_vel = 10.0
        obs_low.extend([-max_box_vel] * 6)
        obs_high.extend([ max_box_vel] * 6)

        # -----------------------------
        # 7) Hand positions (3 + 3)
        # -----------------------------
        max_hand_pos = 2.0
        obs_low.extend([-max_hand_pos] * 3)  # left hand
        obs_high.extend([ max_hand_pos] * 3)

        obs_low.extend([-max_hand_pos] * 3)  # right hand
        obs_high.extend([ max_hand_pos] * 3)

        # -----------------------------
        # 8) Box-to-hand vectors (3 + 3)
        # -----------------------------
        max_rel = 2.0
        obs_low.extend([-max_rel] * 3)  # box_to_left
        obs_high.extend([ max_rel] * 3)

        obs_low.extend([-max_rel] * 3)  # box_to_right
        obs_high.extend([ max_rel] * 3)

        # -----------------------------
        # 9) Target info (3)
        # -----------------------------
        # (target height, current height, height error)
        max_height = 2.0
        obs_low.extend([-max_height, -max_height, -max_height])
        obs_high.extend([ max_height,  max_height,  max_height])

        obs_low = np.array(obs_low, dtype=np.float64)
        obs_high = np.array(obs_high, dtype=np.float64)

        assert obs_low.shape == obs_high.shape, "Low/high bounds shape mismatch!"
        expected_dim = self._calculate_obs_dim()
        assert obs_low.shape[0] == expected_dim, (
            f"Obs bounds dim {obs_low.shape[0]} != obs dim {expected_dim}"
        )

        return obs_low, obs_high



    
    def _find_body_joint(self, body_id):
        """Find the joint ID for a body with a freejoint"""
        for i in range(self.model.njnt):
            if self.model.body_jntadr[body_id] <= i < self.model.body_jntadr[body_id] + self.model.body_jntnum[body_id]:
                return i
        return None
    
    def _calculate_obs_dim(self):
        """Calculate total observation dimensions"""
        dim = 0
        dim += 4  # Left arm qpos
        dim += 4  # Left arm qvel
        dim += 4  # Right arm qpos
        dim += 4  # Right arm qvel
        dim += len(self.controlled_actuators)  # Arm torques/forces
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
        set_body_position(self.model, self.data, "cardboard_box", x=0.34, y=0.0, z=0.76)

        
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
        left_arm_qpos_indices = [22, 23, 24, 25]  # shoulder + elbow + wrist_yaw
        left_arm_qvel_indices = [21, 22, 23, 24]
        left_arm_qpos = self.data.qpos[left_arm_qpos_indices]
        left_arm_qvel = self.data.qvel[left_arm_qvel_indices]
        
        # Right arm state (now 5 DOF with wrist yaw)
        right_arm_qpos_indices = [36, 37, 38, 39]
        right_arm_qvel_indices = [35, 36, 37, 38]
        right_arm_qpos = self.data.qpos[right_arm_qpos_indices]
        right_arm_qvel = self.data.qvel[right_arm_qvel_indices]

        # Arm actuator torques/forces for both arms
        arm_torques = self.data.actuator_force[self.controlled_actuators]

        
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
    
    def check_contact_any_robot_part(self, geom_name):
        """
        Check if ANY robot body (except the box and world) is touching the given geom.
        Used for e.g. 'table_geom' so you can terminate when the robot hits the table.
        """
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Is this contact involving the target geom?
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                # Figure out the "other" geom in the pair
                other_geom = contact.geom2 if contact.geom1 == geom_id else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]

                # Exclude box and world
                if other_body != self.box_body_id and other_body != 0:
                    return True

        return False

    def calculate_reward(self, action: np.ndarray) -> float:
        """
        Reward for Unitree G1 box grasping.

        Priorities, in order:
        1. Move hands near the box (approach).
        2. Make multiple finger contacts with the box (grip),
           but only count as a good grip if it's on the sides.
        3. Lift the box up (lift).
        4. Hold it near target height, stably (no throwing).
        5. Move smoothly (low joint velocity & small actions).
        """

        reward = 0.0
        debug = {}

        # --------------------------------------------------------------
        # Basic weights
        # --------------------------------------------------------------
        w_approach    = 7.0    # get hands to the box
        w_grip        = 3.0    # make multiple contacts
        w_lift        = 8.0    # lift the box
        w_hold        = 7.0    # stable hold near target
        w_action      = 0.003  # stronger penalty on big actions
        w_fail        = 5.0    # penalize dropping box

        w_bad_grip    = 8.0    # penalty if contacts but not side aligned
        w_joint_vel   = 0.10   # penalize fast arm joint velocities
        w_side        = 5.0          # reward for good side alignment

        # stability / overshoot
        w_overshoot   = 20.0
        target_band   = 0.15
        overshoot_tol = 0.05
        side_alignment_threshold = 0.1

        # --------------------------------------------------------------
        # BOX STATE (height + progress)
        # --------------------------------------------------------------
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]

        current_height = float(box_pos[2])
        start_height   = float(self.initial_box_pos[2])
        target_height  = float(self.target_lift_height)

        height_gain  = current_height - start_height
        total_needed = max(target_height - start_height, 1e-6)
        height_progress = float(np.clip(height_gain / total_needed, 0.0, 1.0))

        debug["height_progress"] = height_progress
        debug["current_height"] = current_height
        debug["target_height"] = target_height

        # Box linear velocity
        box_lin_vel = self.data.qvel[self.box_qvel_start:self.box_qvel_start+3]
        box_ang_vel = self.data.qvel[self.box_qvel_start+3:self.box_qvel_start+6]
        ang_speed = float(np.linalg.norm(box_ang_vel))

        vertical_speed = abs(float(box_lin_vel[2]))
        lateral_speed  = float(np.linalg.norm(box_lin_vel[:2]))
        debug["vertical_speed"] = vertical_speed
        debug["lateral_speed"] = lateral_speed

        # --------------------------------------------------------------
        # HAND POSITIONS
        # --------------------------------------------------------------
        left_thumb_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
        left_index_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_0_link")
        left_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link")

        right_thumb_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
        right_index_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
        right_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link")

        left_hand_pos = (
            self.data.xpos[left_thumb_id] +
            self.data.xpos[left_index_id] +
            self.data.xpos[left_middle_id]
        ) / 3.0

        right_hand_pos = (
            self.data.xpos[right_thumb_id] +
            self.data.xpos[right_index_id] +
            self.data.xpos[right_middle_id]
        ) / 3.0

        hands_center = 0.5 * (left_hand_pos + right_hand_pos)

        # --------------------------------------------------------------
        # 1. APPROACH (XY distance of hand-center to box center)
        # --------------------------------------------------------------
        box_xy   = box_pos[:2]
        hands_xy = hands_center[:2]

        dist_xy = float(np.linalg.norm(hands_xy - box_xy))
        approach_term = np.exp(-dist_xy / 0.3)

        reward += w_approach * approach_term
        debug["dist_xy"] = dist_xy
        debug["approach_term"] = approach_term
        debug["approach_contrib"] = w_approach * approach_term

        # --------------------------------------------------------------
        # 2. SIDE ALIGNMENT (pre-grasp geometry)
        # --------------------------------------------------------------
       
        left_align  = self._side_alignment_reward(self.left_thumb_tip_body_id,  self.left_index_tip_body_id)
        right_align = self._side_alignment_reward(self.right_thumb_tip_body_id, self.right_index_tip_body_id)

        side_align = min(left_align, right_align)  # forces both to align
        reward += w_side * side_align


        debug["side_align"] = side_align
        debug["side_align_contrib"] = w_side * side_align

        # --------------------------------------------------------------
        # 3. GRIP: contacts, but *gated* by side alignment
        # --------------------------------------------------------------
        left_contacts = sum(
            1 for body in self.left_finger_bodies
            if self.check_contact(body, "box_geom")
        )
        right_contacts = sum(
            1 for body in self.right_finger_bodies
            if self.check_contact(body, "box_geom")
        )

        total_contacts = left_contacts + right_contacts

        # --- IMPORTANT: define "grip" as a *bilateral* contact, not a single touch.
        # Use the weaker side (min) so PPO can't cheat with one-hand bumps.
        bilateral_contacts = min(left_contacts, right_contacts)

        # Normalize by the smaller finger set size so grip_frac∈[0,1] is meaningful.
        max_per_hand = 2
        grip_frac = float(np.clip(bilateral_contacts / max_per_hand, 0.0, 1.0))

        # side_score ≈ 0 if we're on top; closer to 1.0 for good side alignment
        side_score = side_align
        side_factor = 0.2 + 0.8 * side_score  # in [0.2, 1.0]

        grip_reward = w_grip * grip_frac * side_factor
        reward += grip_reward

        # Only call it a "good grip" if BOTH hands touch AND side alignment decent
        has_side = side_align > side_alignment_threshold
        has_grip = (left_contacts >= 1) and (right_contacts >= 1) and has_side

        # Penalize contacts that are not side-aligned (encourages approaching from sides)
        bad_grip_pen = 0.0
        if total_contacts > 0 and not has_side:
            bad_grip_pen = w_bad_grip * (1.0 - side_align)
            reward -= bad_grip_pen

        # Penalize unilateral touches (prevents "one-hand slap" strategies)
        unilateral_pen = 0.0
        if has_side and ((left_contacts > 0) ^ (right_contacts > 0)):
            unilateral_pen = 0.5 * w_bad_grip
            reward -= unilateral_pen

        debug["left_contacts"] = left_contacts
        debug["right_contacts"] = right_contacts
        debug["total_contacts"] = total_contacts
        debug["bilateral_contacts"] = bilateral_contacts
        debug["grip_frac"] = grip_frac
        debug["side_score"] = side_score
        debug["side_factor"] = side_factor
        debug["grip_contrib"] = grip_reward
        debug["has_side"] = float(has_side)
        debug["has_grip"] = float(has_grip)
        debug["bad_grip_pen"] = bad_grip_pen
        debug["unilateral_pen"] = unilateral_pen

        # --------------------------------------------------------------
        # 4. TOP-CONTACT PENALTY (discourage pressing on the lid)
        # --------------------------------------------------------------
        # Use contact *points* (not just body COM offsets), and include BOTH hands.
        # This term should fire when the robot is "mashing down" on the box top.
        if not hasattr(self, "_box_geom_id"):
            self._box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        box_geom_id = self._box_geom_id

        box_center_z = float(self.data.xpos[self.box_body_id][2])
        top_z = box_center_z + 0.8 * float(self.box_half_height)  # was 0.9; too strict / rarely triggers

        top_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == box_geom_id) or (c.geom2 == box_geom_id):
                # c.pos is the contact point in world coordinates
                if float(c.pos[2]) > top_z:
                    top_contact = True
                    break

        top_contact_pen = 0.0
        if top_contact and total_contacts > 0:
            top_contact_pen = 5.0  # stronger than before
            reward -= top_contact_pen

        debug["top_contact_pen"] = top_contact_pen

        # --------------------------------------------------------------
        # 5. LIFT + STABLE HOLD (only if we have a *good* grip)
        # --------------------------------------------------------------
        # NOTE: In early training, reaching the target height band is rare.
        # To avoid a "dead" stability signal, we compute stability whenever has_grip is true
        # and scale it by height_progress when below the target band.
        stability_term = 0.0
        overshoot = 0.0
        overshoot_pen = 0.0
        vel_pen = 0.0
        hold_contrib = 0.0
        lift_contrib = 0.0

        height_error = current_height - target_height
        debug["height_error"] = height_error

        if has_grip:
            # Lift shaping (still gated by bilateral grip)
            lift_contrib = w_lift * height_progress
            reward += lift_contrib

            # Stability shaping (fires as soon as we have a bilateral grip)
            vel_pen = 0.3 * vertical_speed + 1.0 * lateral_speed
            stability_term = float(np.exp(-(vel_pen ** 2)))

            # If near the target height, reward being still strongly;
            # otherwise provide a smaller shaping reward that grows with lift progress.
            if abs(height_error) <= target_band:
                hold_contrib = w_hold * stability_term
            else:
                hold_contrib = 0.3 * w_hold * stability_term * height_progress

            reward += hold_contrib

            debug["vel_pen"] = vel_pen
            debug["stability_term"] = stability_term
            debug["hold_contrib"] = hold_contrib
            debug["lift_contrib"] = lift_contrib

            # Overshoot penalty (only matters once lifting works)
            overshoot = max(0.0, current_height - (target_height + overshoot_tol))
            if overshoot > 0.0:
                overshoot_pen = w_overshoot * overshoot
                reward -= overshoot_pen

            box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]
            box_rot = np.zeros(9)
            mujoco.mju_quat2Mat(box_rot, box_quat)
            box_rot_mat = box_rot.reshape(3, 3)

            world_z = np.array([0.0, 0.0, 1.0])
            box_z = box_rot_mat[:, 2]
            tilt_cos = abs(np.dot(box_z, world_z))   # 1.0 = perfectly upright

            upright_weight = 8.0
            upright_bonus = upright_weight * float(tilt_cos) * (0.2 + 0.8 * height_progress)
            reward += upright_bonus

            debug["tilt_cos"] = tilt_cos
            debug["ang_speed"] = ang_speed
            debug["upright_bonus"] = upright_bonus
            debug['ang_speed_pen'] = 1.0 * ang_speed * (0.5 + 0.5 * height_progress)

            reward -= 1.0 * ang_speed * (0.5 + 0.5 * height_progress)
            reward += 8.0 * tilt_cos * height_progress

        debug["overshoot"] = overshoot
        debug["overshoot_pen"] = overshoot_pen

        # 6. ACTION & JOINT-VELOCITY PENALTIES (smoothness)
        # --------------------------------------------------------------
        action_norm_sq = float(np.dot(action, action))
        action_pen = w_action * action_norm_sq
        reward -= action_pen

        # Penalize high joint velocities on both arms
        left_arm_qvel_indices  = [21, 22, 23, 24]
        right_arm_qvel_indices = [35, 36, 37, 38]
        left_qvel  = self.data.qvel[left_arm_qvel_indices]
        right_qvel = self.data.qvel[right_arm_qvel_indices]
        joint_vel_norm = float(np.linalg.norm(np.concatenate([left_qvel, right_qvel])))
        joint_vel_pen = w_joint_vel * joint_vel_norm
        reward -= joint_vel_pen

        debug["action_norm_sq"] = action_norm_sq
        debug["action_pen"] = action_pen
        debug["joint_vel_norm"] = joint_vel_norm
        debug["joint_vel_pen"] = joint_vel_pen

        # --------------------------------------------------------------
        # 7. FAIL: box falls below its starting height
        # --------------------------------------------------------------
        fail_pen = 0.0
        if current_height < start_height - 0.05:
            fail_pen = w_fail
            reward -= fail_pen

        debug["fail_pen"] = fail_pen
        debug["total_reward"] = reward

        # --------------------------------------------------------------
        # 8. SANITY: "firing rate" for sparse terms (helps debug reward shaping)
        # --------------------------------------------------------------
        fire_stability = 1.0 if stability_term > 1e-9 else 0.0
        fire_top       = 1.0 if debug.get("top_contact_pen", 0.0) > 1e-9 else 0.0
        fire_overshoot = 1.0 if debug.get("overshoot_pen", 0.0) > 1e-9 else 0.0

        # Exponential moving averages so you can see whether terms ever activate.
        if not hasattr(self, "_fire_ema"):
            self._fire_ema = {"stability": 0.0, "top": 0.0, "overshoot": 0.0}
        beta = 0.99
        self._fire_ema["stability"] = beta * self._fire_ema["stability"] + (1.0 - beta) * fire_stability
        self._fire_ema["top"]       = beta * self._fire_ema["top"]       + (1.0 - beta) * fire_top
        self._fire_ema["overshoot"] = beta * self._fire_ema["overshoot"] + (1.0 - beta) * fire_overshoot

        debug["stability_fire"] = fire_stability
        debug["top_fire"] = fire_top
        debug["overshoot_fire"] = fire_overshoot
        debug["stability_fire_ema"] = float(self._fire_ema["stability"])
        debug["top_fire_ema"] = float(self._fire_ema["top"])
        debug["overshoot_fire_ema"] = float(self._fire_ema["overshoot"])

        # Optional console print every N steps (uncomment if you want)
        # if getattr(self, "current_step", 0) % 2000 == 0:
        #     print(f"[reward firing] stab={self._fire_ema['stability']:.3f} top={self._fire_ema['top']:.3f} over={self._fire_ema['overshoot']:.3f}")
        self.last_reward_debug = debug
        return float(reward)
    
    def terminate(self):
        """
        STRICT ONE-SHOT TERMINATION (but robust to contact flicker):
        - Must acquire a bilateral grasp for K_ACQUIRE consecutive steps.
        - After grasp acquired, if bilateral contact is lost for K_LOSS consecutive steps → FAIL.
        - If it lifts above target while still in bilateral contact → SUCCESS.
        - Safety violations → FAIL (with slightly relaxed box constraints before grasp).
        """

        # -----------------------------
        # Config (tune these)
        # -----------------------------
        Fmin = 2.0          # per-hand force threshold for "meaningful" contact
        K_ACQUIRE = 3       # consecutive steps required to acquire grasp
        K_LOSS = 10         # consecutive steps without bilateral contact to count as loss

        # Pre-grasp tolerances (more forgiving so training can explore)
        XY_TOL_PRE  = 0.25  # was 0.12
        TILT_DEG_PRE = 55   # was 35

        # Post-grasp tolerances (stricter)
        XY_TOL_POST  = 0.12
        TILT_DEG_POST = 35

        # How many consecutive steps box must violate tilt before terminating
        K_TILT = 5

        # -----------------------------
        # Box state
        # -----------------------------
        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]
        current_height = float(box_pos[2])

        # -----------------------------
        # 1) Force-based bilateral contact (per-hand)
        # -----------------------------
        left_force  = sum(self.get_contact_force(b, "box_geom") for b in self.left_finger_bodies)
        right_force = sum(self.get_contact_force(b, "box_geom") for b in self.right_finger_bodies)
        both_touching = (left_force > Fmin) and (right_force > Fmin)

        # -----------------------------
        # 2) Update persistence FIRST (so it actually works)
        # -----------------------------
        if both_touching:
            self.both_touch_steps += 1
        else:
            self.both_touch_steps = 0

        # Acquire grasp only after K consecutive steps
        if (not self.had_bilateral_contact) and (self.both_touch_steps >= K_ACQUIRE):
            self.had_bilateral_contact = True
            self.term_reason = "grasp_acquired"
            # Do NOT terminate here; just mark state.

        # Track loss only after grasp acquired
        if self.had_bilateral_contact and (not both_touching):
            self.grasp_lost_steps += 1
        else:
            self.grasp_lost_steps = 0

        # If grasp was acquired and then lost persistently → FAIL
        if self.had_bilateral_contact and (self.grasp_lost_steps >= K_LOSS):
            self.term_reason = "grasp_lost_persistent"
            return True

        # -----------------------------
        # 3) SUCCESS: lifted above target while grasping NOW
        # -----------------------------
        if self.had_bilateral_contact and both_touching and (current_height >= float(self.target_lift_height)):
            self.term_reason = "success_lifted"
            return True

        # -----------------------------
        # 4) Safety / failure checks
        # -----------------------------

        # Robot-table contact
        if self.check_contact_any_robot_part("table_geom"):
            self.term_reason = "robot_touches_table"
            return True

        # Arm–arm collision
        if self.check_arm_self_collision():
            self.term_reason = "arm_self_collision"
            return True

        # Robot falls (note: you’re reading qpos[2] as torso height; make sure that’s really torso z)
        torso_height = float(self.data.qpos[2])
        if torso_height < 0.6:
            self.term_reason = "robot_fallen"
            return True

        # Box dropped too low (slightly more forgiving than -0.01 if needed)
        if current_height < float(self.initial_box_pos[2]) - 0.02:
            self.term_reason = "box_dropped"
            return True

        # Box moved too far sideways (relaxed pre-grasp, strict post-grasp)
        xy_dist = float(np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2]))
        xy_tol = XY_TOL_POST if self.had_bilateral_contact else XY_TOL_PRE
        if xy_dist > xy_tol:
            self.term_reason = "box_moved_away"
            return True

        # Tilt too much (relaxed pre-grasp, strict post-grasp, plus persistence)
        box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]
        box_rot = np.zeros(9)
        mujoco.mju_quat2Mat(box_rot, box_quat)
        box_rot_mat = box_rot.reshape(3, 3)

        world_z = np.array([0.0, 0.0, 1.0])
        box_z = box_rot_mat[:, 2]
        tilt_cos = abs(float(np.dot(box_z, world_z)))

        tilt_deg_tol = TILT_DEG_POST if self.had_bilateral_contact else TILT_DEG_PRE
        tilt_cos_tol = float(np.cos(np.radians(tilt_deg_tol)))

        if not hasattr(self, "tilt_violation_steps"):
            self.tilt_violation_steps = 0

        if tilt_cos < tilt_cos_tol:
            self.tilt_violation_steps += 1
        else:
            self.tilt_violation_steps = 0

        if self.tilt_violation_steps >= K_TILT:
            self.term_reason = "box_tilted"
            return True

        # Max steps
        if self.current_step >= self.max_episode_steps:
            self.term_reason = "max_steps_exceeded"
            return True

        self.term_reason = "none"
        return False


    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""

        self.both_touch_steps = 0
        self.grasp_lost_steps = 0
        self.had_bilateral_contact = False
        self.tilt_violation_steps = 0
        self.term_reason = "none"


        # reset moving-away tracking per episode
        self.min_left_hand_dist = 1000
        self.min_right_hand_dist = 1000

        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset to initial keyframe
        self._setup_initial_state()

        box_x_offset = np.random.uniform(0, 0)
        #box_y_offset = np.random.uniform(0, 0.2)

        set_body_position(self.model, self.data, "cardboard_box", x=0.34 + box_x_offset, y=0.0, z=0.76)
        mujoco.mj_forward(self.model, self.data)
        self.initial_box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3].copy()
        self.target_lift_height = self.initial_box_pos[2] + 0.3

        # --- compute and store initial hand distances to box ---
        left_thumb_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
        left_index_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_0_link")
        left_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link")

        right_thumb_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
        right_index_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
        right_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link")

        left_hand_pos = (self.data.xpos[left_thumb_id] +
                         self.data.xpos[left_index_id] +
                         self.data.xpos[left_middle_id]) / 3.0

        right_hand_pos = (self.data.xpos[right_thumb_id] +
                          self.data.xpos[right_index_id] +
                          self.data.xpos[right_middle_id]) / 3.0

        self.initial_left_hand_dist  = np.linalg.norm(left_hand_pos  - self.initial_box_pos)
        self.initial_right_hand_dist = np.linalg.norm(right_hand_pos - self.initial_box_pos)
        
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
        self.data.ctrl[self.controlled_actuators] = action

    
        # Lock lower body
        self.data.ctrl[self.locked_actuators] = self.standing_ctrl[self.locked_actuators]
        
        # Lock wrists at fixed angles (palms facing inward)
        # self.data.ctrl[self.left_wrist_locked] = self.standing_ctrl[self.left_wrist_locked]
        # self.data.ctrl[self.right_wrist_locked] = self.standing_ctrl[self.right_wrist_locked]
        
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
            'termination_reason': getattr(self, "term_reason", "not_terminated"),
        }
                
        # Attach reward breakdown for logging/debugging
        if hasattr(self, "last_reward_debug"):
            info.update(self.last_reward_debug)

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
    
    def _has_contact_between(self, geom_ids_a, geom_ids_b) -> bool:
        """
        Return True if any contact exists between any geom in A and any geom in B.
        """
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in geom_ids_a and g2 in geom_ids_b) or (g2 in geom_ids_a and g1 in geom_ids_b):
                return True
        return False
    
    def _thumb_box_contact(self) -> bool:
        return self._has_contact_between(self.left_thumb_geom_ids, [self.box_geom_id])

    def _index_box_contact(self) -> bool:
        return self._has_contact_between(self.left_index_geom_ids, [self.box_geom_id])
        
    def _side_alignment_reward(self, thumb_tip_body_id, index_tip_body_id) -> float:
        box_pos   = self.data.xpos[self.box_body_id]
        thumb_pos = self.data.xpos[thumb_tip_body_id]
        index_pos = self.data.xpos[index_tip_body_id]

        thumb_offset = thumb_pos - box_pos
        index_offset = index_pos - box_pos

        top_thresh = 0.85 * self.box_half_height
        if thumb_offset[2] > top_thresh or index_offset[2] > top_thresh:
            return 0.0

        y_sep   = abs(thumb_offset[1] - index_offset[1])
        y_score = np.clip(y_sep / (2 * self.box_half_depth), 0.0, 1.0)

        sigma = 0.08
        x_center = 0.5 * (thumb_offset[0] + index_offset[0])
        x_score = np.exp(- (x_center ** 2) / (2 * sigma ** 2))

        return float(np.clip(y_score * x_score, 0.0, 1.0))


    
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
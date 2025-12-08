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
        self.w_move_away = 2.0
        
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
        # Left arm
        obs_low.extend(self.model.actuator_ctrlrange[self.left_arm_actuators, 0])
        obs_high.extend(self.model.actuator_ctrlrange[self.left_arm_actuators, 1])
        # Right arm
        obs_low.extend(self.model.actuator_ctrlrange[self.right_arm_actuators, 0])
        obs_high.extend(self.model.actuator_ctrlrange[self.right_arm_actuators, 1])
        
        # Arm velocity limits (keep slow!)
        max_arm_vel = 1.0  # rad/s - adjust this to control speed of robot's arms
        obs_low.extend([-max_arm_vel] * 10)  # 5 DOF x 2 arms
        obs_high.extend([max_arm_vel] * 10)

        # Arm torque/force observation (unbounded or large bounds)
        obs_low.extend([-np.inf] * 10)
        obs_high.extend([np.inf] * 10)
        
        # Box state (position, orientation, velocity) - unbounded
        obs_low.extend([-np.inf] * 13)
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
        dim += 10 # Arm actuator torques (5 DOF x 2 arms)
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
    
    def calculate_reward(self, action: np.ndarray) -> float:
        """
        Simplified reward focused on:
          1. Mirrored arms (shoulder roll & pitch symmetry)
          2. Bringing hands to the sides of the box (via shoulder roll)
          3. Lifting the box up (via shoulder pitch)

        Extra things like grip force shaping, finger balance, detailed
        vertical alignment, etc. are intentionally removed.
        """
        reward = 0.0

        # -----------------------------
        # Weights (tune as needed)
        # -----------------------------
        w_symmetry_roll  = 4.0   # left roll ≈ -right roll
        w_symmetry_pitch = 4.0   # left pitch ≈ right pitch
        w_reach          = 6.0   # hands to box sides
        w_lift           = 10.0  # lifting the box
        w_contact_gate   = 4.0   # encourage bilateral contact
        w_smooth         = 0.002 # very small action penalty
        w_fail           = 5.0   # soft penalty for dropping box
        w_wrist_yaw      = 8.0 

        # --------------------------------------------------------------
        # BOX STATE
        # --------------------------------------------------------------
        box_pos  = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]
        box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]

        current_height = box_pos[2]
        start_height   = self.initial_box_pos[2]
        target_height  = self.target_lift_height

        height_gain  = current_height - start_height
        total_needed = max(target_height - start_height, 1e-6)
        height_progress = np.clip(height_gain / total_needed, 0.0, 1.0)

        # --------------------------------------------------------------
        # HAND POSITIONS (averaged over thumb / index / middle)
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

        # --------------------------------------------------------------
        # 1. ARM SYMMETRY (directly targets shoulder roll & pitch)
        # --------------------------------------------------------------
        # Assumes you have these indices defined in __init__:
        #   self.left_shoulder_roll_qpos_idx
        #   self.right_shoulder_roll_qpos_idx
        #   self.left_shoulder_pitch_qpos_idx
        #   self.right_shoulder_pitch_qpos_idx

        left_roll  = float(self.data.qpos[self.left_shoulder_roll_qpos_idx])
        right_roll = float(self.data.qpos[self.right_shoulder_roll_qpos_idx])

        left_pitch  = float(self.data.qpos[self.left_shoulder_pitch_qpos_idx])
        right_pitch = float(self.data.qpos[self.right_shoulder_pitch_qpos_idx])

        # Roll: want left ≈ -right  → (left + right) ≈ 0
        roll_sym_error = left_roll + right_roll
        roll_sym_term  = np.exp(- (roll_sym_error ** 2) / (2.0 * (0.15 ** 2)))  # ~0.15 rad tolerance
        reward += w_symmetry_roll * roll_sym_term

        # Pitch: want left ≈ right → (left - right) ≈ 0
        pitch_sym_error = left_pitch - right_pitch
        pitch_sym_term  = np.exp(- (pitch_sym_error ** 2) / (2.0 * (0.15 ** 2)))
        reward += w_symmetry_pitch * pitch_sym_term

        # --------------------------------------------------------------
        # 1b. WRIST YAW TARGETS: left ≈ -0.6, right ≈ +0.6 (or 6.0 if you mean that)
        # --------------------------------------------------------------
        left_wrist_yaw  = float(self.data.qpos[self.left_wrist_yaw_qpos_idx])
        right_wrist_yaw = float(self.data.qpos[self.right_wrist_yaw_qpos_idx])

        # Desired targets
        target_left_yaw  = -0.6
        target_right_yaw =  0.6   # change to 6.0 if you literally mean 6 rad

        # Gaussian-shaped rewards around the target angles
        sigma = 0.15  # tolerance in radians

        left_yaw_err  = left_wrist_yaw  - target_left_yaw
        right_yaw_err = right_wrist_yaw - target_right_yaw

        left_yaw_term  = np.exp(- (left_yaw_err  ** 2) / (2.0 * sigma ** 2))
        right_yaw_term = np.exp(- (right_yaw_err ** 2) / (2.0 * sigma ** 2))

        wrist_yaw_reward = 0.5 * (left_yaw_term + right_yaw_term)
        reward += w_wrist_yaw * wrist_yaw_reward


        # --------------------------------------------------------------
        # 2. REACHING: HANDS TO BOX SIDES (driven mostly by shoulder roll)
        # --------------------------------------------------------------
        # We want left hand on +Y side and right hand on -Y side at box center height
        xNudge = 0.0   # can bias forward/back if you want
        target_left  = box_pos + np.array([xNudge,  self.box_half_depth, 0.0])
        target_right = box_pos + np.array([xNudge, -self.box_half_depth, 0.0])

        left_dist  = np.linalg.norm(left_hand_pos  - target_left)
        right_dist = np.linalg.norm(right_hand_pos - target_right)

        reach_scale      = 0.4  # characteristic distance
        left_reach_term  = np.exp(-left_dist  / reach_scale)
        right_reach_term = np.exp(-right_dist / reach_scale)
        reach_reward     = 0.5 * (left_reach_term + right_reach_term)

        reward += w_reach * reach_reward

        # --------------------------------------------------------------
        # 3. SIMPLE CONTACT GATE (so it learns to actually pinch the box)
        # --------------------------------------------------------------
        left_finger_touching  = any(self.check_contact(b, "box_geom") for b in self.left_finger_bodies)
        right_finger_touching = any(self.check_contact(b, "box_geom") for b in self.right_finger_bodies)
        both_touching         = left_finger_touching and right_finger_touching

        # small bonus once it manages bilateral contact
        if both_touching and not self.had_bilateral_contact:
            reward += w_contact_gate * 2.0
            self.had_bilateral_contact = True

        # tiny dense reward for maintaining bilateral contact
        if both_touching:
            reward += w_contact_gate * 0.2

        # --------------------------------------------------------------
        # 4. LIFT REWARD (primarily driven by shoulder pitch)
        # --------------------------------------------------------------
        # Only reward lifting if it's actually holding the box with both hands
        if both_touching:
            reward += w_lift * height_progress

            # extra bonus on/above target
            if current_height >= target_height:
                reward += w_lift * 3.0

        # --------------------------------------------------------------
        # 5. SMALL ACTION PENALTY (keep it tame, but still allowed to move)
        # --------------------------------------------------------------
        action_norm_sq = float(np.dot(action, action))
        reward -= w_smooth * action_norm_sq

        # --------------------------------------------------------------
        # 6. VERY SIMPLE "FAIL" SIGNALS (not about style, just disasters)
        # --------------------------------------------------------------
        # Box fell significantly below start
        if current_height < start_height - 0.1:
            reward -= w_fail

        return float(reward)

    
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


    def terminate(self):
        """
        STRICT ONE-SHOT TERMINATION:
        - Robot has exactly ONE chance to pick up the box.
        - If it ever loses the grasp after getting it → FAIL.
        - If it lifts above target → SUCCESS.
        - All safety violations → FAIL.
        """

        box_pos = self.data.qpos[self.box_qpos_start:self.box_qpos_start+3]
        current_height = box_pos[2]

        # -----------------------------
        # 1. Check finger contact
        # -----------------------------
        left_touching = any(
            self.check_contact(body, "box_geom") for body in self.left_finger_bodies
        )
        right_touching = any(
            self.check_contact(body, "box_geom") for body in self.right_finger_bodies
        )
        both_touching = left_touching and right_touching

        # First-time grasp
        if both_touching and not self.had_bilateral_contact:
            self.had_bilateral_contact = True

        # Lost grasp after having it once → FAIL
        if self.had_bilateral_contact and not both_touching:
            return True

        # SUCCESS: lifted above target while grasping
        if self.had_bilateral_contact and current_height >= self.target_lift_height:
            return True
        
        # -----------------------------
        # NEW: moved away without ever grasping → FAIL
        # -----------------------------
        if not self.had_bilateral_contact and self.current_step > 20:
            # compute current hand distances to box
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

            left_dist_now  = np.linalg.norm(left_hand_pos  - box_pos)
            right_dist_now = np.linalg.norm(right_hand_pos - box_pos)

            margin = 0.10  # how much further than start we tolerate

            if (left_dist_now  > self.initial_left_hand_dist  + margin or
                right_dist_now > self.initial_right_hand_dist + margin):
                return True

        # Box dropped too low
        if current_height < self.initial_box_pos[2] - 0.05:
            return True

        # Box moved too far sideways
        xy_dist = np.linalg.norm(box_pos[:2] - self.initial_box_pos[:2])
        if xy_dist > 0.12:
            return True

        # Tilt too much
        box_quat = self.data.qpos[self.box_qpos_start+3:self.box_qpos_start+7]
        box_rot = np.zeros(9)
        mujoco.mju_quat2Mat(box_rot, box_quat)
        box_rot_mat = box_rot.reshape(3, 3)

        world_z = np.array([0.0, 0.0, 1.0])
        box_z = box_rot_mat[:, 2]
        tilt_cos = abs(np.dot(box_z, world_z))

        if tilt_cos < np.cos(np.radians(25)):
            return True

        # Arm–arm collision
        if self.check_arm_self_collision():
            return True

        # Robot touches table
        if self.check_contact_any_robot_part("table_geom"):
            return True

        # Robot falls
        torso_height = self.data.qpos[2]
        if torso_height < 0.6:
            return True

        # Max steps
        if self.current_step >= self.max_episode_steps:
            return True

        return False

    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""

        self.had_bilateral_contact = False

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
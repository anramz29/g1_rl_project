import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

"""
MuJoCo Robot Viewer with Pelvis Welded to World + Keyframe Saving + Arm Mirroring + Two Boxes

This script loads a robot with two boxes and welds the pelvis to the world so it CANNOT fall over,
even when arms move. The RIGHT arm automatically mirrors the LEFT arm.
Press 's' + Enter to save current pose as a keyframe.
"""

def input_listener(save_flag):
    """Listen for keyboard input in a separate thread"""
    while True:
        user_input = input()
        if user_input.lower() == 's':
            save_flag['save'] = True

def save_keyframe_to_file(data, filename="saved_keyframe.txt"):
    """Save current qpos and ctrl to a text file in keyframe format"""
    
    # Get current state
    qpos = data.qpos.copy()
    ctrl = data.ctrl.copy()
    
    # Open file and write
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SAVED KEYFRAME - Copy this into your XML file\n")
        f.write("="*60 + "\n\n")
        
        # Write the keyframe XML format
        f.write('<key name="custom_pose"\n')
        f.write('  qpos="\n')
        
        # Write qpos values (formatted nicely)
        # Format: values in groups for readability
        f.write("  ")
        for i, val in enumerate(qpos):
            f.write(f"{val:.6f} ")
            # New line every 7 values for readability
            if (i + 1) % 7 == 0 and i < len(qpos) - 1:
                f.write("\n  ")
        f.write('"\n')
        
        # Write ctrl values
        f.write('  ctrl="\n')
        f.write("  ")
        for i, val in enumerate(ctrl):
            f.write(f"{val:.6f} ")
            # New line every 7 values for readability
            if (i + 1) % 7 == 0 and i < len(ctrl) - 1:
                f.write("\n  ")
        f.write('"\n')
        f.write('/>\n\n')
        
        # Also write human-readable version
        f.write("="*60 + "\n")
        f.write("HUMAN READABLE FORMAT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of qpos elements: {len(qpos)}\n")
        f.write(f"Number of ctrl elements: {len(ctrl)}\n\n")
        
        f.write("qpos values:\n")
        for i, val in enumerate(qpos):
            f.write(f"  [{i:2d}] {val:+.6f}\n")
        
        f.write("\nctrl values:\n")
        for i, val in enumerate(ctrl):
            f.write(f"  [{i:2d}] {val:+.6f}\n")
    
    print(f"\n{'='*60}")
    print(f"✓ KEYFRAME SAVED to: {filename}")
    print(f"{'='*60}")
    print(f"qpos elements: {len(qpos)}")
    print(f"ctrl elements: {len(ctrl)}")
    print(f"\nCopy the keyframe definition from the file into your XML!")
    print(f"{'='*60}\n")

def mirror_left_arm_to_right(data):
    """Mirror left arm configuration to right arm"""
    
    # Left arm actuator indices (from XML): 15-21
    # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
    left_arm_ctrl_ids = [15, 16, 17, 18, 19, 20, 21]
    
    # Right arm actuator indices (from XML): 29-35
    right_arm_ctrl_ids = [29, 30, 31, 32, 33, 34, 35]
    
    # Left hand actuator indices: 22-28
    left_hand_ctrl_ids = [22, 23, 24, 25, 26, 27, 28]
    
    # Right hand actuator indices: 36-42
    right_hand_ctrl_ids = [36, 37, 38, 39, 40, 41, 42]
    
    # Mirror signs for arm joints
    # You may need to adjust these based on testing
    # Typical: shoulder_pitch (same), shoulder_roll (flip), shoulder_yaw (flip), 
    #          elbow (same), wrist_roll (flip), wrist_pitch (same), wrist_yaw (flip)
    arm_mirror_signs = np.array([1, -1, -1, 1, -1, 1, -1])
    
    # Mirror signs for hand joints (may need adjustment)
    hand_mirror_signs = np.array([1, -1, -1, -1, -1, -1, -1])
    
    # Get left arm control values
    left_arm_ctrl = data.ctrl[left_arm_ctrl_ids]
    left_hand_ctrl = data.ctrl[left_hand_ctrl_ids]
    
    # Mirror to right arm
    right_arm_ctrl = left_arm_ctrl * arm_mirror_signs
    right_hand_ctrl = left_hand_ctrl * hand_mirror_signs
    
    # Apply to right arm
    data.ctrl[right_arm_ctrl_ids] = right_arm_ctrl
    data.ctrl[right_hand_ctrl_ids] = right_hand_ctrl

def set_body_position(model, data, body_name, x, y, z):
    """
    Set the position of a body with a freejoint in MuJoCo.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body to move (e.g., "cardboard_box")
        x, y, z: New position coordinates
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        # Find body's joint (assumes it has a freejoint)
        joint_id = None
        for i in range(model.njnt):
            if model.body_jntadr[body_id] <= i < model.body_jntadr[body_id] + model.body_jntnum[body_id]:
                joint_id = i
                break
        
        if joint_id is not None:
            # Get qpos address for joint
            qpos_addr = model.jnt_qposadr[joint_id]
            
            # Set new position (x, y, z, quat_w, quat_x, quat_y, quat_z)
            data.qpos[qpos_addr + 0] = x
            data.qpos[qpos_addr + 1] = y
            data.qpos[qpos_addr + 2] = z
            
            # Keep orientation as identity quaternion (no rotation)
            data.qpos[qpos_addr + 3] = 1.0   # quat_w
            data.qpos[qpos_addr + 4] = 0.0   # quat_x
            data.qpos[qpos_addr + 5] = 0.0   # quat_y
            data.qpos[qpos_addr + 6] = 0.0   # quat_z
            
            # IMPORTANT: Call mj_forward to update dependent quantities
            mujoco.mj_forward(model, data)
            
            print(f"✓ {body_name} repositioned to: [{x:.2f}, {y:.2f}, {z:.2f}]")
            return True
        else:
            print(f"⚠ Could not find joint for {body_name}")
            return False
            
    except Exception as e:
        print(f"⚠ Could not reposition {body_name}: {e}")
        return False

def main():
    # Path to your XML file with two boxes
    xml_path = r"/Users/adrian/g1_picknplace/Upper_Body/g1_two_boxes_custom_keyframes_friction.xml"
    
    # Select a graphics backend for the viewer
    os.environ.setdefault("MUJOCO_GL", "glfw")
    
    # Load model and create data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Start from the "stand" keyframe
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("✓ Robot set to 'stand' keyframe")
    except Exception as e:
        print(f"⚠ Could not load 'stand' keyframe: {e}")
    
    # Position the two boxes
    set_body_position(model, data, "table_box", x=0.7, y=0.0, z=0.3)
    set_body_position(model, data, "cardboard_box", x=0.38, y=0.0, z=0.7)
    
    # Manually fix floating base
    try:
        base_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if base_joint_id >= 0:
            qpos_addr = model.jnt_qposadr[base_joint_id]
            fixed_base_qpos = data.qpos[qpos_addr:qpos_addr+7].copy()
            print(f"✓ Base fixed at position: {fixed_base_qpos[:3]}")
            use_manual_fix = True
        else:
            print("⚠ Floating base joint not found")
            use_manual_fix = False
    except:
        print("⚠ Could not set up manual base fixing")
        use_manual_fix = False
    
    # Define which actuators to lock (legs + waist)
    locked_actuator_ids = list(range(0, 15))
    standing_ctrl = data.ctrl.copy()
    
    # Set up save flag for thread communication
    save_flag = {'save': False}
    
    # Start input listener thread
    input_thread = threading.Thread(target=input_listener, args=(save_flag,), daemon=True)
    input_thread.start()
    
    print("="*60)
    print("MUJOCO ROBOT VIEWER - ARM MIRRORING + KEYFRAME SAVER + BOXES")
    print("="*60)
    print("Instructions:")
    print("- Pelvis is WELDED - robot CANNOT fall over")
    print("- Two boxes are visible (table_box and cardboard_box)")
    print("- LEFT arm is FREE - use MuJoCo sliders to control it")
    print("- RIGHT arm MIRRORS left arm automatically")
    print("- Right-click → 'Control' to access joint sliders")
    print("- Only move LEFT arm joints - right will follow")
    print("- Position robot as desired, then type 's' + ENTER to save")
    print("="*60)
    print("\n⚠ NOTE: If mirroring looks wrong, you may need to adjust")
    print("  the mirror_signs array in the code (testing required)")
    print("\nType 's' and press ENTER to save current pose...")
    print()
    
    # Launch viewer and step the simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        
        while viewer.is_running():
            # Check if user wants to save
            if save_flag['save']:
                save_keyframe_to_file(data)
                save_flag['save'] = False  # Reset flag
                print("Ready for next save. Type 's' + ENTER to save again...")
            
            # Lock lower body controls
            data.ctrl[locked_actuator_ids] = standing_ctrl[locked_actuator_ids]
            
            # Mirror left arm to right arm BEFORE physics step
            mirror_left_arm_to_right(data)
            
            # Manually fix base position
            if use_manual_fix:
                data.qpos[qpos_addr:qpos_addr+7] = fixed_base_qpos
                data.qvel[0:6] = 0
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            step_count += 1
    
    print("\n" + "="*60)
    print("Simulation ended.")

if __name__ == "__main__":
    main()
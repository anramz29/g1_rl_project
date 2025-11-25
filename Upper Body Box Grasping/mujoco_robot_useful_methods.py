import mujoco


def load_keyframe(model, data, keyframe_name): # load a keyframe to the robot by name
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        #print(f"✓ Robot set to '{keyframe_name}' keyframe")
    except Exception as e:
        print(f"⚠ Could not load '{keyframe_name}' keyframe: {e}")
        print("  Loading default 'stand' keyframe instead...")
        try:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        except:
            print("⚠ Could not load 'stand' keyframe either")



def set_body_position(model, data, body_name, x, y, z): #set a body to a certain position (mainly the cardboard box and the table box)
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
            
            # Update model
            mujoco.mj_forward(model, data)
            
            #print(f"✓ {body_name} repositioned to: [{x:.2f}, {y:.2f}, {z:.2f}]")
            return True
        else:
            print(f"⚠ Could not find joint for {body_name}")
            return False
            
    except Exception as e:
        print(f"⚠ Could not reposition {body_name}: {e}")
        return False
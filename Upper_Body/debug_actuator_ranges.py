import mujoco
from g1_box_grasp_env_both_arms_adrian import G1BoxGraspEnv

if __name__ == "__main__":
    env = G1BoxGraspEnv(render_mode=None)

    print("=== Actuator force ranges ===")
    for i in range(env.model.nu):
        name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
        )
        fr = env.model.actuator_forcerange[i]
        print(f"{i:2d} | {name:35s} | min={fr[0]:8.3f} max={fr[1]:8.3f}")
    print("=== End actuator force ranges ===")
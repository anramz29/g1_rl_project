import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardDebugCallback(BaseCallback):
    """
    Logs pieces of env.info (reward breakdown) to TensorBoard.
    Works with vectorized envs (uses mean over envs).
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None or len(infos) == 0:
            return True

        if self.n_calls % self.log_freq != 0:
            return True

        def log_mean_if_present(key: str, tb_name: str | None = None):
            vals = [info[key] for info in infos if key in info]
            if len(vals) > 0:
                if tb_name is None:
                    tb_name = key
                self.logger.record(tb_name, float(np.mean(vals)))

        def log_rate_if_present(key: str, tb_name: str | None = None, thresh: float = 0.5):
            """Logs fraction of envs where info[key] > thresh (good for booleans stored as 0/1)."""
            vals = [info[key] for info in infos if key in info]
            if len(vals) > 0:
                if tb_name is None:
                    tb_name = key + "_rate"
                self.logger.record(tb_name, float(np.mean(np.array(vals) > thresh)))

        # ====== 1. High-level performance ======
        log_mean_if_present("total_reward",    "debug/total_reward")
        log_mean_if_present("height_progress", "debug/height_progress")
        log_mean_if_present("current_height",  "debug/current_height")
        log_mean_if_present("height_error",    "debug/height_error")

        # ====== 2. Grasp quality ======
        log_mean_if_present("approach_term",   "debug/approach_term")
        log_mean_if_present("side_align",      "debug/side_align")
        log_rate_if_present("has_side",        "debug/has_side_rate")
        log_mean_if_present("grip_frac",       "debug/grip_frac")
        log_mean_if_present("total_contacts",  "debug/total_contacts")
        log_mean_if_present("left_contacts",   "debug/left_contacts")
        log_mean_if_present("right_contacts",  "debug/right_contacts")
        log_rate_if_present("has_grip",        "debug/has_grip_rate")
        log_mean_if_present("top_contact_pen", "debug/top_contact_pen")
        log_mean_if_present("bad_grip_pen",    "debug/bad_grip_pen")

        # ====== 3. Lift & stability ======
        log_mean_if_present("stability_term",  "debug/stability_term")
        log_mean_if_present("overshoot_pen",   "debug/overshoot_pen")
        log_mean_if_present("vertical_speed",  "debug/vertical_speed")
        log_mean_if_present("lateral_speed",   "debug/lateral_speed")

        # --- NEW: anti-tip / post-lift stability terms ---
        log_mean_if_present("tilt_cos",        "debug/tilt_cos")
        log_mean_if_present("ang_speed",       "debug/ang_speed")
        log_mean_if_present("upright_bonus",   "debug/upright_bonus")
        log_mean_if_present("ang_pen",         "debug/ang_pen")

        # ====== 4. Smoothness / control ======
        log_mean_if_present("action_norm_sq",  "debug/action_norm_sq")
        log_mean_if_present("action_pen",      "debug/action_pen")
        log_mean_if_present("joint_vel_norm",  "debug/joint_vel_norm")
        log_mean_if_present("joint_vel_pen",   "debug/joint_vel_pen")

        # ====== 5. Termination reason rates (super helpful) ======
        # You must put info["term_reason"] = self.term_reason in env.step()
        reasons = [info.get("term_reason", None) for info in infos]
        reasons = [r for r in reasons if r is not None]

        if len(reasons) > 0:
            # log the fraction of envs reporting each reason at this step
            unique = sorted(set(reasons))
            for r in unique:
                self.logger.record(f"debug/term_reason_rate/{r}", float(np.mean([x == r for x in reasons])))

        return True

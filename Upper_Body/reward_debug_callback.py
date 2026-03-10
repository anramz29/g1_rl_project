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
        # infos from vec env
        infos = self.locals.get("infos", None)
        if infos is None or len(infos) == 0:
            return True

        # only log every N calls to reduce spam
        if self.n_calls % self.log_freq != 0:
            return True

        # Helper: average a key over all envs that have it
        def log_mean_if_present(key: str, tb_name: str | None = None):
            vals = [info[key] for info in infos if key in info]
            if len(vals) > 0:
                if tb_name is None:
                    tb_name = key
                self.logger.record(tb_name, float(np.mean(vals)))

        # ==== log whatever you care about ====
        log_mean_if_present("total_reward",   "debug/total_reward")
        log_mean_if_present("height_progress","debug/height_progress")
        log_mean_if_present("current_height", "debug/current_height")
        log_mean_if_present("target_height",  "debug/target_height")
        log_mean_if_present("height_error",   "debug/height_error")

        log_mean_if_present("grip_frac",      "debug/grip_frac")
        log_mean_if_present("total_contacts", "debug/total_contacts")

        log_mean_if_present("stability_term", "debug/stability_term")
        log_mean_if_present("overshoot",      "debug/overshoot")
        log_mean_if_present("overshoot_pen",  "debug/overshoot_pen")

        log_mean_if_present("action_norm_sq", "debug/action_norm_sq")
        log_mean_if_present("action_pen",     "debug/action_pen")

        log_mean_if_present("vertical_speed", "debug/vertical_speed")
        log_mean_if_present("lateral_speed",  "debug/lateral_speed")

        # you can add more keys from last_reward_debug if you want

        return True

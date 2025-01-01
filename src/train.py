import os
import pickle
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient

from stable_baselines3 import DQN

def make_training_env():
    """
    Creates a gym environment and normalize observation and rewards.
    """
    env = Monitor(HIVPatient(domain_randomization=True))
    env = TimeLimit(env, max_episode_steps=200)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=1000.0)
    return vec_env

class ProjectAgent:
    def __init__(self):
        self.model = None

        normalize_path = "vec_normalize_stats.pkl"
        if not os.path.exists(normalize_path):
            normalize_path = "src/" + normalize_path
        with open(normalize_path, "rb") as f:
            stats = pickle.load(f)
        self.obs_mean = stats.obs_rms.mean
        self.obs_var  = stats.obs_rms.var


    def act(self, observation, use_random=False):
        if self.model is None:
            return 0

        std = np.sqrt(self.obs_var + 1e-8)
        normalized_obs = (observation - self.obs_mean) / std
        action, _ = self.model.predict(normalized_obs, deterministic=(not use_random))
        return int(action)

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)

    def load(self, model_path=None):
        model_path = "ppo_hiv.zip"
        if not os.path.exists(model_path):
            model_path = "src/" + model_path
            if not os.path.exists(model_path):
                print("Model file not found. Agent will act randomly.")
                return
                
        self.model = PPO.load(model_path, device="cpu")
        print(f"Loaded model from {model_path}")


if __name__ == "__main__":
    train_env = make_training_env()

    policy_kwargs = {"net_arch": [256, 256]}
    model = PPO(
        policy="MlpPolicy",
        env=train_env,  
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,
        gamma=0.9999,
        n_steps=4096,
        batch_size=128,
        gae_lambda=0.95,
        verbose=1,
        device="cuda",
        tensorboard_log="./tb_logs",
    )

    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_HIV")

    model.save("ppo_hiv.zip")
    print("Training complete! Model saved to ppo_hiv.zip.")

    # Save VecNormalize statistics so we can reload them later to normalize the observations.
    train_env.save("vec_normalize_stats.pkl")
    print("Normalization stats saved to vec_normalize_stats.pkl.")

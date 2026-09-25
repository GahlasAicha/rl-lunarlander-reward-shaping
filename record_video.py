import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

def record(model_path, video_folder, video_name):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_name,
                       episode_trigger=lambda x: True)
    model = PPO.load(model_path)
    observation, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


record("ppo_lunarlander", "videos", "baseline")
record("ppo_lunarlander_shaped", "videos", "shaped")

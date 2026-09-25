import gymnasium as gym 
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3",render_mode="human")
model = PPO.load("ppo_lunarlander_shaped",env=env)
observation , info = env.reset()
for _ in range(1000):
    action,_states = model.predict(observation , deterministic=True)
    observation ,reward ,terminated , truncated , info = env.step(action)
    if terminated or truncated :
        obersvation , info = env.reset()
env.close()
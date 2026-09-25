import gymnasium as gym
from stable_baselines3 import PPO 

env = gym.make("LunarLander-v3")
 # create agent ppo , MlpPolicy = reseau de neurones utiliser pur decider des   ctions 
model = PPO("MlpPolicy",env, verbose=1)
model.learn (total_timesteps=200000)

model.save("ppo_LunarLander")

env.close()
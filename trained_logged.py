import gymnasium as gym 
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor 
import os 

os.makedirs("logs",exist_ok=True)

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self,env,time_penalty =0.0):
        super().__init__(env)
        self.time_penalty = time_penalty
    def step(self,action):
        observation ,reward,terminated, truncated ,info=self.env.step(action)
        reward -= self.time_penalty
        return observation , reward , terminated , truncated,info
def train (log_name ,time_penalty):
        env = gym.make("LunarLander-v3")
        env = RewardShapingWrapper(env, time_penalty=time_penalty)
        env = Monitor(env, filename=f"logs/{log_name}")

        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=200000)
        model.save(f"ppo_lunarlander_{log_name}")
        env.close()
train("baseline",time_penalty=0.0)
train("shaped",time_penalty=0.3)
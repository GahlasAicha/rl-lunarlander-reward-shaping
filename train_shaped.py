import gymnasium as gym 
from stable_baselines3 import PPO 

class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper qui modifie la recompence pour decourager le flottement : 
    on ajoute une petite penalite a chaque  pas de temps , ce qui pousse
    l'agent a atterir rapidement pplutot que de rester en vol indefiniment .
    """
    def __init__(self, env, time_penalty= 0.3):
        super().__init__(env)
        self.time_penalty = time_penalty

    def step(self ,action):
        observation , reward , terminated , truncated, info= self.env.step(action)
        reward -= self.time_penalty 
        return observation , reward,terminated,truncated ,info


env =  gym.make ("LunarLander-v3")
env = RewardShapingWrapper(env,time_penalty=0.3)

model = PPO("MlpPolicy" , env , verbose=1)
model.learn(total_timesteps=200000)
model.save("ppo_lunarlander_shaped")

env.close()

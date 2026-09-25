import gymnasium as gym 
from stable_baselines3 import PPO

def evaluate (model_path , n_episodes= 50):
    env = gym.make("LunarLander-v3")
    model = PPO.load(model_path)

    rewards = []
    successes = 0
    craches = 0

    for _ in range(n_episodes):
        observation , info = env.reset()
        done = False 
        total_reward = 0

        while not done:
            action , _states = model.predict(observation,deterministic=True)
            observation , reward , terminated , truncated , info = env.step(action)
            total_reward += reward 
            done = terminated or truncated 

        rewards.append(total_reward)
        if total_reward >= 200:
            successes +=1
        elif total_reward <= -100:
            craches +=1
    env.close()
    mean_reward = sum(rewards) / len(rewards)
    return mean_reward, successes ,craches
        
for name ,path in [
    ("Baseline", "ppo_lunarlander"),
    ("Avec reward shaping", "ppo_lunarlander_shaped"),
]:
    mean_r , succ , crash = evaluate(path)
    print(f"\n{name}")
    print(f"  Recompense moyenne (score brut du jeu): {mean_r:.1f}")
    print(f"  Atterrissages reussis: {succ}/50")
    print(f"  Crashs: {crash}/50")
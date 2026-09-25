import gymnasium as gym 
env =gym.make("LunarLander-v3", render_mode = "human")

observation ,info = env.reset()
# observation :ce que l'agent voit , c des chiffres decrivant la postion , la vitesse , angle du module 

for _ in range (500):
    action = env.action_space.sample()
    observation , reward , terminated , truncated , info = env.step(action)
    if terminated or truncated :
         observation , info = env.reset()
env.close()






    


















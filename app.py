import streamlit as st
import gymnasium as gym 
from stable_baselines3 import PPO


st.set_page_config(
    page_title="RL Lunar Lander",
    page_icon=":rocket:",
    layout="wide"
)

@st.cache_data
def evaluate(model_path , n_episodes=50):
    env = gym.make("LunarLander-v3")
    model = PPO.load(model_path)

    rewards = []
    successes = 0
    crashes = 0

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
            crashes +=1
    env.close()
    mean_reward = sum(rewards) / len(rewards)
    success_rate = successes/n_episodes * 100
    crashes_rate = crashes/n_episodes * 100
    return mean_reward, success_rate, crashes_rate



st.title("Agent RL : atterissage lunaire avec PPO")

st.markdown("""
Ce projet compare deux agents entraines avec **PPO** (Stable-Baselines3) sur l'environnement
**LunarLander-v3** : un agent baseline, et un agent avec **reward shaping** (penalite de temps)
pour corriger un comportement de "vol stationnaire".
""")

with st.spinner("Evaluation des modeles en cours (50 episodes chacun)..."):
    baseline_reward, baseline_success, baseline_crash = evaluate("ppo_lunarlander_baseline")
    shaped_reward, shaped_success, shaped_crash = evaluate("ppo_lunarlander_shaped")

reward_delta = (shaped_reward - baseline_reward) / abs(baseline_reward) * 100
success_delta = shaped_success - baseline_success

st.header("Resultats chiffres")
col1, col2, col3 = st.columns(3)
col1.metric("Recompense moyenne", f"{shaped_reward:.1f}", f"{reward_delta:+.0f}% vs baseline")
col2.metric("Atterrissages reussis", f"{shaped_success:.0f}%", f"{success_delta:+.0f} points vs baseline")
col3.metric("Crashs", f"{shaped_crash:.0f}%", f"{shaped_crash - baseline_crash:+.0f} points vs baseline")

st.caption(f"Baseline : recompense {baseline_reward:.1f}, reussite {baseline_success:.0f}%, crash {baseline_crash:.0f}% (50 episodes, mode deterministe)")

st.header("Courbe d'apprentissage")
st.image("comparison_curve.png", caption="Score brut du jeu, moyenne glissante sur 20 episodes")

st.header("Videos de l'agent")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Baseline (sans shaping)")
    st.video("videos/baseline-episode-0.mp4")
with col2:
    st.subheader("Avec reward shaping")
    st.video("videos/shaped-episode-0.mp4")
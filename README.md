# RL LunarLander - Reward Shaping

Reinforcement learning agent (PPO) trained on LunarLander-v3, with diagnosis and correction of a reward hacking behavior through reward shaping.

**[View live demo](https://rl-lunarlander-reward-shaping-hrjfduynqmxbjxkf5ebkqw.streamlit.app/)**

## The Problem

A first PPO agent trained for 200,000 timesteps converges to a final score near 0, far below the environment's "solved" threshold (+200). Key observation: average episode length kept increasing throughout training (up to ~680 steps) without a matching increase in reward.

**Diagnosis:** the agent learned to exploit a flaw in the reward structure. Crashing costs -100, but hovering indefinitely costs almost nothing. The agent therefore learned to "hover" instead of landing, a classic reward hacking pattern in RL.

## The Fix: Reward Shaping

Added a -0.3 penalty per timestep through a Gymnasium wrapper, forcing the agent to land quickly rather than stall in the air.

## Results

Evaluated over 50 episodes in deterministic mode, using the raw game score (penalty removed for a fair comparison):

| Metric | Baseline | With reward shaping |
|---|---|---|
| Mean reward | 73.9 | 183.7 |
| Successful landings | 4% (2/50) | 72% (36/50) |
| Crashes | 0% | 0% |

Success rate improved 18x, with no increase in crash rate. The gain comes from the agent actually committing to land, not from riskier behavior.

![Comparison curve](comparison_curve_fixed.png)

## Project Structure

- train.py: baseline PPO training
- train_shaped.py: PPO training with reward shaping
- train_logged.py: logged version for the comparison curve
- compare.py: quantitative evaluation of both models
- plot_comparaison.py: generates the learning curve
- record_video.py: records before/after videos
- app.py: Streamlit dashboard

## Tech Stack

- Gymnasium: simulation environment (LunarLander-v3)
- Stable-Baselines3: PPO implementation
- PyTorch: underlying compute engine
- Streamlit: interactive demo dashboard

## Reproducing the Project

Install dependencies with pip install -r requirements.txt

Then, in order:
- python train_logged.py to train both models with logging
- python compare.py to evaluate and compare both models
- python plot_comparaison.py to generate the comparison curve
- streamlit run app.py to launch the dashboard locally

## Author

GahlasAicha

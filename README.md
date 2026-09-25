# RL LunarLander - Reward Shaping

Agent de reinforcement learning (PPO) entraine sur LunarLander-v3, avec diagnostic et correction d'un comportement de "reward hacking" via reward shaping.

**[Voir la demo en ligne](https://rl-lunarlander-reward-shaping-hrjfduynqmxbjxkf5ebkqw.streamlit.app/)**

## Le probleme

Un premier agent PPO entraine sur 200 000 pas de temps atteint un score final proche de 0, bien en dessous du seuil de "resolution" du jeu (+200). L'observation cle : la duree moyenne des episodes augmente au fil de l'entrainement (jusqu'a ~680 pas) sans que le score progresse en consequence.

**Diagnostic :** l'agent a appris a exploiter une faille du systeme de recompense. Se crasher coute -100, mais rester indefiniment en vol stationnaire ne coute presque rien. L'agent a donc appris a "flotter" plutot qu'a atterrir, un comportement classique de reward hacking en RL.

## La correction : reward shaping

Ajout d'une penalite de -0.3 par pas de temps via un wrapper Gymnasium, pour forcer l'agent a atterrir rapidement plutot que de temporiser.

## Resultats

Evaluation sur 50 episodes, en mode deterministe, score brut du jeu (penalite retiree pour comparaison equitable) :

| Metrique | Baseline | Avec reward shaping |
|---|---|---|
| Recompense moyenne | 73.9 | 183.7 |
| Atterrissages reussis | 4 pourcent (2/50) | 72 pourcent (36/50) |
| Crashs | 0 pourcent | 0 pourcent |

Le taux de reussite est multiplie par 18, sans augmentation du taux de crash. Le gain vient du fait que l'agent se decide a atterrir, pas d'une prise de risque accrue.

![Courbe de comparaison](comparison_curve_fixed.png)

## Structure du projet

- train.py : entrainement PPO baseline
- train_shaped.py : entrainement PPO avec reward shaping
- train_logged.py : version avec logging pour la courbe de comparaison
- compare.py : evaluation chiffree des deux modeles
- plot_comparaison.py : generation de la courbe d'apprentissage
- record_video.py : enregistrement des videos avant et apres
- app.py : dashboard Streamlit

## Stack technique

- Gymnasium : environnement de simulation (LunarLander-v3)
- Stable-Baselines3 : implementation de l'algorithme PPO
- PyTorch : moteur de calcul sous-jacent
- Streamlit : dashboard interactif de demonstration

## Reproduire le projet

Installer les dependances avec pip install -r requirements.txt

Puis dans l'ordre :
- python train_logged.py pour entrainer les 2 modeles avec logging
- python compare.py pour evaluer et comparer les 2 modeles
- python plot_comparaison.py pour generer la courbe de comparaison
- streamlit run app.py pour lancer le dashboard en local

## Auteur

GahlasAicha

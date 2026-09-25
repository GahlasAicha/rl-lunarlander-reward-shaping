import pandas as pd 
import matplotlib.pyplot as plt 

def load_log (path, time_penalty) :
    # skiprows igonre la premiere ligne 
    df = pd.read_csv(path , skiprows=1)
    df["episode"] = range(len(df))
    df["r_raw"] = df["r"] + time_penalty * df["l"]
    df["reward_smooth"] = df["r_raw"].rolling(window=20 , min_periods =1).mean()
    return df
baseline = load_log("logs/baseline.monitor.csv", time_penalty=0.0)
shaped = load_log("logs/shaped.monitor.csv", time_penalty=0.3)


plt.figure(figsize=(10,6))
plt.plot(baseline["episode"], baseline["reward_smooth"], label="Baseline (sans shaping)", color="tab:red")
plt.plot(shaped["episode"], shaped["reward_smooth"], label="Avec reward shaping", color="tab:green")

plt.xlabel("Episode")
plt.ylabel("Recompense moyenne (fenetre de 20 episodes)")
plt.title("Comparaison de l'apprentissage : baseline vs reward shaping")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


plt.savefig("comparison_curve.png", dpi=150)
plt.show()
print("Graphique sauvegarde dans comparison_curve.png")
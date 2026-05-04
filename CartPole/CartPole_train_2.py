import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

alpha = 0.1
gamma = 0.99
epsilon_start = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
TAU_VALUES = [-3.0, -2.0, -1, 0.0, 1, 2, 3, 4]
MAX_EPISODES = 20000
N_RUNS     = 3
q_table = {}


def get_q(q_table, state, action):
    return q_table.get((state, action), 0.0)


def smooth(rewards, window=50):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def discretize(obs):
    bins = [
        np.linspace(-2.4, 2.4, 15),   
        np.linspace(-4, 4, 10),     
        np.linspace(-0.418, 0.418, 25), 
        np.linspace(-8, 8, 25)      
    ]   
    
    state = []
    for i in range(len(obs)):
        state.append(np.digitize(obs[i], bins[i]))
    
    return tuple(state)

def train(tau_value):
    tau = tau_value
    env = gym.make("CartPole-v1") 
    q_table = {}
    epsilon = epsilon_start
    episode_rewards = []
    for i in range(MAX_EPISODES):
        obs, _ = env.reset()
        state = discretize(obs)
        
        total_reward = 0

        for t in range(500):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = [get_q(q_table, state, a) for a in range(2)]
                action = np.argmax(q_vals)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)

            done = terminated or truncated

            best_next = max([get_q(q_table, next_state, a) for a in range(2)])
            old_q = get_q(q_table, state, action)

            td_error = reward + gamma * best_next - old_q
            if td_error >= tau:
                q_table[(state, action)] = old_q + alpha * td_error

            state = next_state
            total_reward += reward
            if(reward >= 475):
                break
            if done:
                break
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
    env.close()
    return episode_rewards

results = {}   
aucl    = {}  

for tau in TAU_VALUES:
    print("Start for tau = ", tau)
    runs = []
    for run in range(N_RUNS):
        rewards = train(tau)
        runs.append(rewards)
    results[tau] = runs
    aucl[tau] = [np.trapezoid(r) for r in runs]
    mean_aucl = np.mean(aucl[tau])
    print("Mean of AUCL is:", mean_aucl)

# ============================================================
# PLOT 1 — AUCL vs TAU (bar chart with error bars)
# ============================================================
tau_vals   = TAU_VALUES
aucl_means = [np.mean(aucl[t]) for t in tau_vals]
aucl_stds  = [np.std(aucl[t])  for t in tau_vals]
 
# find best tau
best_idx = int(np.argmax(aucl_means))
colors   = ['#378ADD' if i != best_idx else '#3B9B3F' for i in range(len(tau_vals))]
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CartPole-v1 — Clipped TD Error Sweep', fontsize=14, fontweight='bold')
 
ax1 = axes[0]
bars = ax1.bar(
    [str(t) for t in tau_vals],
    aucl_means,
    yerr=aucl_stds,
    color=colors,
    capsize=5,
    edgecolor='white',
    linewidth=0.5
)
ax1.axhline(y=aucl_means[tau_vals.index(0.0)], color='gray',
            linestyle='--', linewidth=1, label='τ=0 baseline')
ax1.set_xlabel('τ (clipping threshold)')
ax1.set_ylabel('AUCL (area under learning curve)')
ax1.set_title('AUCL vs τ')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)
for i, (m, s) in enumerate(zip(aucl_means, aucl_stds)):
    ax1.text(i, m + s + 50, f'{m:,.0f}', ha='center', fontsize=8, color='#333')
 
# ============================================================
# PLOT 2 — SMOOTHED LEARNING CURVES for select tau values
# ============================================================
ax2 = axes[1]
highlight_taus = [-2.0, -1.0, 0.0, 1, 2, 3]  # subset for readability
cmap = plt.cm.coolwarm
colors_lc = [cmap(i / (len(highlight_taus) - 1)) for i in range(len(highlight_taus))]
 
for tau, col in zip(highlight_taus, colors_lc):
    avg_rewards = np.mean(results[tau], axis=0)
    smoothed    = smooth(avg_rewards, window=50)
    episodes    = np.arange(len(smoothed))
    ax2.plot(episodes, smoothed, label=f'τ={tau:+.1f}', color=col, linewidth=1.5)
 
ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward (smoothed, window=50)')
ax2.set_title('Learning Curves (select τ values)')
ax2.legend(fontsize=9)
 
plt.tight_layout()
plt.savefig('cartpole_tau_sweep.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to cartpole_tau_sweep.png")
plt.show()
 
# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n── Summary ──────────────────────────────")
print(f"{'tau':>6}  {'mean AUCL':>12}  {'std':>8}")
print("─" * 32)
for t in tau_vals:
    m = np.mean(aucl[t])
    s = np.std(aucl[t])
    marker = " ← best" if t == tau_vals[best_idx] else ""
    print(f"{t:>6.1f}  {m:>12,.0f}  {s:>8,.0f}{marker}")
 
# save AUCL data for later use
with open("aucl_results.pkl", "wb") as f:
    pickle.dump({"tau_values": TAU_VALUES, "aucl": aucl, "rewards": results}, f)
print("\nRaw results saved to aucl_results.pkl")


            
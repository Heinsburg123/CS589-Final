import gymnasium as gym
import numpy as np
import random
import pickle

train_env = gym.make("Taxi-v4")

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
q_table = {}

def get_q(state, action):
    return q_table.get((state, action), 0.0)

mean_reward = []
episode = 0

while True:
    # obs is already an integer state — no discretize call needed
    obs, _ = train_env.reset()
    state = obs
    total_reward = 0

    for t in range(200):
        # ε-greedy policy
        if random.random() < epsilon:
            action = train_env.action_space.sample()
        else:
            q_vals = [get_q(state, a) for a in range(6)]   # 6 actions in Taxi
            action = int(np.argmax(q_vals))

        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        next_state = next_obs                               # already discrete
        done = terminated or truncated

        best_next = max(get_q(next_state, a) for a in range(6))
        old_q     = get_q(state, action)

        q_table[(state, action)] = old_q + alpha * (
            reward + gamma * best_next - old_q
        )

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    mean_reward.append(total_reward)

    if episode >= 50000:
        with open("q_table_taxi.pkl", "wb") as f:
            pickle.dump(q_table, f)
        print("Q-table saved (episode limit reached).")
        break

    if len(mean_reward) == 100:
        avg = sum(mean_reward) / 100
        print(f"Episode {episode}, average reward: {avg:.2f}")

        if avg > 7:
            with open("q_table_taxi.pkl", "wb") as f:
                pickle.dump(q_table, f)
            print("Q-table saved — environment solved!")
            break
        mean_reward.clear()

    episode += 1
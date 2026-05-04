import gymnasium as gym
import numpy as np
import pickle

test_env = gym.make("Taxi-v4", render_mode="human")

with open("q_table_taxi.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_q(state, action):
    return q_table.get((state, action), 0.0)


obs, _ = test_env.reset()
state = obs

for _ in range(200):
    q_vals = [get_q(state, a) for a in range(6)]
    action = int(np.argmax(q_vals))

    obs, reward, terminated, truncated, _ = test_env.step(action)
    state = obs

    if terminated or truncated:
        obs, _ = test_env.reset()
        state = obs

test_env.close()
import gymnasium as gym
import numpy as np
import pickle

test_env = gym.make("CliffWalking-v1", render_mode="human")

with open("q_table_cliffwalking.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_q(state, action):
    return q_table.get((state, action), 0.0)

# no discretize needed — CliffWalking state is already an integer (0–47)

obs, _ = test_env.reset()
state = obs

for _ in range(200):
    q_vals = [get_q(state, a) for a in range(4)]
    action = int(np.argmax(q_vals))

    obs, reward, terminated, truncated, _ = test_env.step(action)
    state = obs

    if terminated or truncated:
        obs, _ = test_env.reset()
        state = obs

test_env.close()
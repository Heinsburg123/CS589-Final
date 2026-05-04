import gymnasium as gym
import numpy as np
import pickle

test_env = gym.make("MountainCar-v0", render_mode="human")

with open("q_table_mountaincar.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_q(state, action):
    return q_table.get((state, action), 0.0)

def discretize(obs):
    bins = [
        np.linspace(-1.2,  0.6,  30),  
        np.linspace(-0.07, 0.07, 30),  
    ]
    return tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))

obs, _ = test_env.reset()
state = discretize(obs)

for i in range(200):
    q_vals = [get_q(state, a) for a in range(3)]
    action = np.argmax(q_vals)

    obs, reward, terminated, truncated, _ = test_env.step(action)
    state = discretize(obs)   

    if terminated or truncated:
        break

test_env.close()
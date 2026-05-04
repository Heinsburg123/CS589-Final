import gymnasium as gym
import numpy as np
import random
import gymnasium as gym
import pickle

test_env = gym.make("CartPole-v1", render_mode="human")

with open("q_table_cartpole.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_q(state, action):
    return q_table.get((state, action), 0.0)

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

obs, _ = test_env.reset()
state = discretize(obs)

for _ in range(500):
    q_vals = [get_q(state, a) for a in range(2)]
    action = np.argmax(q_vals)

    obs, reward, terminated, truncated, _ = test_env.step(action)
    state = discretize(obs)

    if terminated or truncated:
        obs, _ = test_env.reset()
test_env.close()
import gymnasium as gym
import numpy as np
import random
import pickle
import math 

train_env = gym.make("MountainCar-v0")

alpha = 0.1
gamma = 0.85
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
q_table = {}

def get_q(state, action):
    return q_table.get((state, action), 0.0)

def discretize(obs):
    bins = [
        np.linspace(-1.2,  0.6,  30),   # position
        np.linspace(-0.07, 0.07, 30),   # velocity
    ]
    return tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))

def shape_reward(obs, next_obs):
    pos, vel = obs
    next_pos, next_vel = next_obs
    height_gain = math.sin(3 * next_pos) - math.sin(3 * pos)
    speed_gain = abs(next_vel) - abs(vel)        
    return height_gain + speed_gain * 10

mean_reward = []
episode = 0

while True:
    obs, _ = train_env.reset()
    state = discretize(obs)
    total_reward = 0

    for t in range(200):   
        if random.random() < epsilon:
            action = train_env.action_space.sample()
        else:
            q_vals = [get_q(state, a) for a in range(3)]  
            action = np.argmax(q_vals)

        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        next_state = discretize(next_obs)
        done = terminated or truncated

        shaped = reward + shape_reward(obs, next_obs)

        best_next = max(get_q(next_state, a) for a in range(3))
        old_q     = get_q(state, action)

        q_table[(state, action)] = old_q + alpha * (
            shaped + gamma * best_next - old_q
        )

        obs = next_obs
        state = next_state
        total_reward += reward  

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    mean_reward.append(total_reward)

    if episode >= 50000:
        with open("q_table_mountaincar.pkl", "wb") as f:
            pickle.dump(q_table, f)
        print("Q-table saved (episode limit reached).")
        break

    if len(mean_reward) == 100:
        avg = sum(mean_reward) / 100
        print(f"Episode {episode}, average reward: {avg:.2f}")

        if avg > -110:
            with open("q_table_mountaincar.pkl", "wb") as f:
                pickle.dump(q_table, f)
            print("Q-table saved — environment solved!")
            break
        mean_reward.clear()

    episode += 1
import gymnasium as gym
import numpy as np
import random
import gymnasium as gym
import pickle

train_env = gym.make("CartPole-v1") 

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
q_table = {}

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

mean_reward = []
episode = 0
while True:

    obs, _ = train_env.reset()
    state = discretize(obs)
    
    total_reward = 0

    for t in range(500):

        # ε-greedy policy
        if random.random() < epsilon:
            action = train_env.action_space.sample()
        else:
            q_vals = [get_q(state, a) for a in range(2)]
            action = np.argmax(q_vals)

        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        next_state = discretize(next_obs)

        done = terminated or truncated

        best_next = max([get_q(next_state, a) for a in range(2)])
        old_q = get_q(state, action)

        q_table[(state, action)] = old_q + alpha * (
            reward + gamma * best_next - old_q
        )

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    mean_reward.append(total_reward)
    if(episode >=20000):
        with open("q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)
        print("Q-table saved.")
        break
    if len(mean_reward) == 100:
        print(f"Episode {episode}, average reward: {sum(mean_reward)/100}")
        if(sum(mean_reward)/100 > 450):
            with open("q_table_cartpole.pkl", "wb") as f:
                pickle.dump(q_table, f)
            print("Q-table saved.")
            break
        mean_reward.clear()
            
    episode += 1
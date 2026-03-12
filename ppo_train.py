import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from block_bootstrap import SimulatedInvestor
from ppo_agent import PPOAgent

# set seed for debugging
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# enable anomaly detection to catch unhandled NaNs at their source
torch.autograd.set_detect_anomaly(True)

# key numbers
ACTION_DIMENSION = 4
STATE_DIMENSION = 8

# set training parameters
NUM_TRAJECTORIES = 5_000
CAUTION = 0       
BETA = 0.04
STARTING_WEALTH = 1_000_000

investor = SimulatedInvestor()
agent = PPOAgent(state_size=STATE_DIMENSION, action_size=ACTION_DIMENSION)
timesteps_per_batch = 400

def get_financial_update(state, action_old, withdraw_old, wealth_old):
    asset_returns = state[:4]
    inflation_rate = state[4]
    withdraw = (1 + inflation_rate) * withdraw_old
    wealth = wealth_old * (1 + np.dot(asset_returns, action_old)) - withdraw
    
    # Scale the reward
    reward = (wealth - wealth_old) / wealth_old
    
    return reward, withdraw, wealth

avg_actions = np.zeros((NUM_TRAJECTORIES, 4), dtype=np.float32)
financial_ruin = np.zeros(NUM_TRAJECTORIES, dtype=np.float32)
terminal_wealth = np.zeros(NUM_TRAJECTORIES, dtype=np.float32)

def train():
    start_time = time.time()
    print('Beginning training...')
    step_ticker = 0

    for i in range(NUM_TRAJECTORIES):
        wealth = STARTING_WEALTH
        withdraw = BETA * STARTING_WEALTH
        action_sum = np.array([0., 0., 0., 0.])

        trajectory_length = investor.generate_time_after_retirement()
        trajectory_states = investor.get_trajectory(trajectory_length)

        for t in range(trajectory_length - 1):
            state = trajectory_states[t]
            new_state = trajectory_states[t + 1]

            if i <= CAUTION:
                action = np.array([0., 0., 0., 1.])
                reward, withdraw, wealth = get_financial_update(new_state, action, withdraw, wealth)
                if wealth <= 0: break
                continue

            action, log_prob, value = agent.get_action_and_value(state)
            action_sum += action
            
            reward, withdraw, wealth = get_financial_update(new_state, action, withdraw, wealth)
            is_done = wealth <= 0

            agent.store_transition(state, action, reward, is_done, log_prob, value)
            step_ticker += 1

            if step_ticker % timesteps_per_batch == 0:
                _, _, next_value = agent.get_action_and_value(new_state)
                agent.update(next_value)

            if is_done:
                print(f'Oh no! Life {i} ends in financial ruin! Ends with wealth {wealth}')
                financial_ruin[i] = 1
                break

            if t == trajectory_length - 2:
                print(f'Life {i} ends with wealth {wealth}')
                terminal_wealth[i] = wealth

        sum_of_action_sums = np.sum(action_sum)
        if sum_of_action_sums > 0:
            avg_actions[i, :] = action_sum / sum_of_action_sums

        if (i + 1) % 25 == 0:
            agent.save_checkpoint()
            print(f'...Saving checkpoint. Time elapsed: {time.time() - start_time:.2f} seconds')

    print(f'TRAINING COMPLETE. Time elapsed: {time.time() - start_time:.2f} seconds.')

train()

# calculate rolling average for financial ruin and terminal wealth
window = 50

fin_ruin_pd = pd.Series(financial_ruin)
roll_fr = fin_ruin_pd.rolling(window=window, min_periods=1).mean()

term_wealth_pd = pd.Series(terminal_wealth)
roll_tw = term_wealth_pd.rolling(window=window, min_periods=1).mean()

legend = ['stocks', 'housing', 'bonds', 'bills']

fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 7))
axes = axes.ravel()

for i, ax in enumerate(axes):
    if i <= 3:
        ax.plot(avg_actions[:, i])
        ax.set_title(f'Avg weight accorded to {legend[i]}')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    if i == 4:
        ax.plot(roll_tw)
        ax.set_title(f'{window}-point rolling terminal wealth')
        ax.grid(True, alpha=0.3)
    if i == 5:
        ax.plot(roll_fr)
        ax.set_title(f'{window}-point prob of financial ruin')
        ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
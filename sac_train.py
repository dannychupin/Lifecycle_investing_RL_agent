import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

from block_bootstrap import SimulatedInvestor
from sac_agent import SoftActorCritic

# I. Seed for debugging
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# II. Determine countries, assets, and macros
ALLOWED_COUNTRIES = ['Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
                     'Denmark', 'Spain', 'Finland', 'France', 'UK',
                     'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
                     'Portugal', 'Sweden', 'USA']  # keep all
ALLOWED_ASSETS = [['eq_tr', 'eq_capgain', 'eq_div_rtn'], ['bond_tr', 'bond_rate']]  # just keep equity and bonds
ALLOWED_MACROS = [['gdp'], ['cpi']]         # always keep cpi last (for inflation calculation)

ACTION_DIMENSION = len(ALLOWED_ASSETS)
OBSERVATION_DIMENSION = len(ALLOWED_ASSETS + ALLOWED_MACROS)
STATE_DIMENSION = OBSERVATION_DIMENSION + 2  # append wealth_t, withdraw_t

"""
IMPORTANT:
    state       = observation + wealth + withdraw
    observation = stock rtn + bond rtn + ... + inflation
    
    SO:
    state[-2] = wealth
    state[-1] = withdraw
    observation[-1] = inflation
    observation[:ACTION_DIMENSION] = asset rtns
"""


# III. Set up agent and simulation

# settings for agent
HIDDEN1_DIMENSION = 30
HIDDEN2_DIMENSION = 30
MAX_MEMORY_SIZE = 5_000  # about 2x the number 3000 of unique state-state transitions
BATCH_SIZE = 10  # make it big enough to lower variance, but not so big that it's maximal

GAMMA = 0.99
H_TARGET = -ACTION_DIMENSION
ALPHA_INITIAL = 1.

LR = 3e-4
LR_ACTOR = LR
LR_CRITIC = LR
LR_ALPHA = LR  # Temperature learning rate
TAU = 0.005  # Update for the two Q_target networks; from SAC papers

# training details
NUM_TRAJECTORIES = 2_000
WAIT = 50  # how many trajectories to wait before starting gradient steps (say 50% of dataset)
BETA = 0.04
STARTING_WEALTH = 1_000

# initialize simulation and agent
investor = SimulatedInvestor(countries=ALLOWED_COUNTRIES,
                             asset_proxy_list=ALLOWED_ASSETS,
                             macro_list=ALLOWED_MACROS)

agent = SoftActorCritic(state_dimension=STATE_DIMENSION,
                        action_dimension=ACTION_DIMENSION,
                        hidden1_dimension=HIDDEN1_DIMENSION,
                        hidden2_dimension=HIDDEN2_DIMENSION,
                        tau=TAU,
                        lr_actor=LR_ACTOR,
                        lr_critic=LR_CRITIC,
                        lr_alpha=LR_ALPHA,
                        H_target=H_TARGET,
                        max_memory_size=MAX_MEMORY_SIZE,
                        batch_size=BATCH_SIZE,
                        alpha_initial=ALPHA_INITIAL,
                        gamma=GAMMA,
                        checkpoint_file_name="SAC_model_weights/checkpoint.pt")


# descriptive metrics for training
avg_action = np.zeros((NUM_TRAJECTORIES, ACTION_DIMENSION), dtype=np.float32)
financial_ruin = np.zeros(NUM_TRAJECTORIES, dtype=np.float32)
terminal_wealth = np.zeros(NUM_TRAJECTORIES, dtype=np.float32)
policy_gn = np.zeros(NUM_TRAJECTORIES, dtype=np.float32)
q1_gn = np.zeros(NUM_TRAJECTORIES, dtype=np.float32)
q2_gn = np.zeros(NUM_TRAJECTORIES, dtype=np.float32)


def observation_to_state(observation, wealth, withdraw):

    return np.concatenate((observation, [wealth], [withdraw]), dtype=np.float32)


def get_financial_update(observation, action_old, state_old):

    # extract relevant state info for update
    asset_returns = observation[:ACTION_DIMENSION]
    inflation_rate = observation[-1]
    wealth_old = state_old[-2]
    withdraw_old = state_old[-1]

    # update `withdraw_old` to keep up with recent inflation
    withdraw = (1 + inflation_rate) * withdraw_old

    # update `wealth_old` with new returns from `state`, then withdraw
    wealth = wealth_old * (1 + np.dot(asset_returns, action_old)) - withdraw
    reward = wealth - wealth_old

    return reward, wealth, withdraw


def train():
    start_time = time.time()
    print('Beginning training...')

    for i in range(NUM_TRAJECTORIES):
        # initialize wealth and withdraw amount
        wealth = STARTING_WEALTH
        withdraw = BETA * STARTING_WEALTH  # will try to maintain this real amount of consumption
        action_sum = np.zeros(ACTION_DIMENSION)

        # get chain of states
        trajectory_length = investor.generate_time_after_retirement()
        trajectory_observations = investor.get_trajectory(trajectory_length)

        # I. Take actions and update replay buffer (order is SAS'R'D')
        agent.policy.eval()
        with torch.inference_mode():
            for t in range(trajectory_length - 1):

                # extract observation and complete to a state
                observation = trajectory_observations[t]
                state = observation_to_state(observation, wealth, withdraw)

                # take action
                action_dist = agent.policy.forward(torch.from_numpy(state))
                action = action_dist.sample().numpy()
                print(f'On trajectory {i} and time {t}, the sampled action is {action}.')

                action_sum += action

                # make new observation, update withdraw and wealth
                observation_new = trajectory_observations[t + 1]
                reward, wealth_new, withdraw_new = get_financial_update(observation_new, action, state)

                # update is_done
                is_done = False
                if wealth_new <= 0 or t + 1 == trajectory_length - 1:
                    # if either go in the negative or finish the trajectory:
                    is_done = True

                # complete to a new state and update replay buffer
                state_new = observation_to_state(observation_new, wealth_new, withdraw_new)
                agent.memory.store_transition(state, action, state_new, reward, is_done)

                # update wealth and withdraw for next time step
                wealth = wealth_new
                withdraw = withdraw_new

                # at the end of the trajectory: either go broke, or end with positive wealth
                if is_done:
                    terminal_wealth[i] = wealth

                    if wealth < 0:
                        financial_ruin[i] = 1
                        break
        print('')
        print('')

        # II. Try to take gradient step
        if i > WAIT:
            policy_grad_norm, Q1_grad_norm, Q2_grad_norm = agent.update()
            policy_gn[i] = policy_grad_norm
            q1_gn[i] = Q1_grad_norm
            q2_gn[i] = Q2_grad_norm

        # III. Update metrics for the trajectory
        sum_of_action_sums = np.sum(action_sum)
        if sum_of_action_sums > 0:
            avg_action[i, :] = action_sum / sum_of_action_sums

        # save checkpoint every 100 trajectories
        if (i + 1) % 100 == 0:
            agent.save_checkpoint()
            diff = time.time() - start_time
            print(f'Trajectory {i + 1} checkpoint saved. Time elapsed: {diff} seconds')

    time_elapsed = time.time() - start_time
    print(f'TRAINING COMPLETE. Time elapsed: {time_elapsed} seconds.')
    print(f'Average of {time_elapsed/NUM_TRAJECTORIES} seconds / trajectory')


train()

# calculate rolling averages for metrics
window = 30

fin_ruin_pd = pd.Series(financial_ruin)
roll_fr = fin_ruin_pd.rolling(window=window, min_periods=1).mean()

term_wealth_pd = pd.Series(terminal_wealth)
roll_tw = term_wealth_pd.rolling(window=window, min_periods=1).mean()

policy_gn_pd = pd.Series(policy_gn)
roll_policy_gn = policy_gn_pd.rolling(window=window, min_periods=1).mean()
q1_gn_pd = pd.Series(q1_gn)
roll_q1_gn = q1_gn_pd.rolling(window=window, min_periods=1).mean()
q2_gn_pd = pd.Series(q2_gn)
roll_q2_gn = q2_gn_pd.rolling(window=window, min_periods=1).mean()

stock_wt_pd = pd.Series(avg_action[:, 0])
roll_stock_wt = stock_wt_pd.rolling(window=window, min_periods=1).mean()

bond_wt_pd = pd.Series(avg_action[:, 1])
roll_bond_wt = bond_wt_pd.rolling(window=window, min_periods=1).mean()


# plot metrics

fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 7))
axes = axes.ravel()

# avg asset wts
axes[0].plot(roll_stock_wt)
axes[0].set_title(f'Rolling avg weight on stocks')
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3)

axes[1].plot(roll_bond_wt)
axes[1].set_title(f'Rolling avg weight on bonds')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)

axes[2].plot(roll_fr)
axes[2].set_title(f'Rolling prob of financial ruin')
axes[2].grid(True, alpha=0.3)

axes[3].plot(roll_tw)
axes[3].set_title(f'Rolling terminal wealth')
axes[3].grid(True, alpha=0.3)

axes[4].plot(roll_policy_gn)
axes[4].set_title(f'Rolling policy grad norm')
axes[4].grid(True, alpha=0.3)

axes[5].plot(q1_gn_pd)
q2_gn_pd.plot(ax=axes[5], color='red', linestyle='--')
axes[5].set_title(f'Rolling Q1, Q2 grad norm')
axes[5].grid(True, alpha=0.3)


fig.tight_layout()
plt.show()

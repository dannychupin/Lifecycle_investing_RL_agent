"""
    Version: April 4, 2026

    ac = Actor-Critic; nothing "soft" here

    changes from sac_train:
    - cleaned up/overhauled global parameter specification at the beginning, to accord with LaTeX document
    - introduced parameter NUM_TRAJECTORIES_PER_POLICY to modify how many trajectories are unrolled under each policy

    BIG MYSTERY:
    - under 'simple' rewards, policy value should be on the order of 10 (likely between 0 and 20):
            "10-ish years of successful withdrawals"
    - under 'simple' rewards, Q1, Q2 losses (MSE losses) should also be maybe on the order of 10 I think:
            "Q-functions estimate policy values within sqrt{10} rewards of Q-target networks

    ...but they're all HUGE
"""

import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time

from block_bootstrap import SimulatedInvestor
from ac_agent import ActorCritic

### I. Seed for debugging ###

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

### II. Determine countries, assets, and macros ###

# all countries and assets, for reference
ALL_COUNTRIES = ['Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
                 'Denmark', 'Spain', 'Finland', 'France', 'UK',
                 'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
                 'Portugal', 'Sweden', 'USA']

ALL_ASSETS = [['eq_tr', 'eq_capgain', 'eq_div_rtn'],
              ['housing_tr', 'housing_capgain', 'housing_rent_rtn'],
              ['bond_tr', 'bond_rate'],
              ['bill_rate']]

# select allowed countries and assets. Be simple: stocks and bonds for now!
ALLOWED_COUNTRIES = ['Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
                     'Denmark', 'Spain', 'Finland', 'France', 'UK',
                     'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
                     'Portugal', 'Sweden', 'USA']  # keep all

ALLOWED_ASSETS = [['eq_tr', 'eq_capgain', 'eq_div_rtn'],
                  ['bond_tr', 'bond_rate']]

ALLOWED_MACROS = [['cpi']]  # always keep cpi last (for inflation calculation). Here, just keep inflation

ACTION_DIMENSION = len(ALLOWED_ASSETS)

# inflation will be incorporated into real returns, so subtract 1 for eliminating inflation
OBSERVATION_DIMENSION = len(ALLOWED_ASSETS) + len(ALLOWED_MACROS) - 1

# states will be [real returns on assets] + [other real macro returns] + [wealth]
# for us this is 2 + 0 + 1
STATE_DIMENSION = OBSERVATION_DIMENSION + 1

"""
IMPORTANT:
    `observations` are NOMINAL return vectors, including inflation (return on cpi)
    observation[-1] = inflation rate, by convention

    `states` are REAL return vectors:
    state = observation_real + wealth_real
    (so if there are 2 assets, state is 2+1=3 dimensional)

    SO:
    state[-1] = real wealth
    observation[:ACTION_DIMENSION] = real asset rtns

    To convert an observation to a state, first produce a `real observation` = observation[:-2] - observation[-1]
"""

### III. Set up agent and simulation ###

# 1) DEFAULT hyperparameters (DO NOT CHANGE OR ERASE)
HIDDEN1_DIMENSION = 8
HIDDEN2_DIMENSION = 8

DISTRIBUTION_TYPE = 'gaussian'      # 'gaussian' is better than 'dirichlet'
ACTIVATIONS = {'relu': nn.ReLU(inplace=True),
               'tanh': nn.Tanh()}
ACTIVATION_NAME = 'relu'            # 'relu' is better than 'tanh'
ACTIVATION = ACTIVATIONS[ACTIVATION_NAME]

SHOULD_CLIP_GRADS = True
CLIP_VALUE = 10.0

# annotated with notation from LaTeX document
AVG_TRAJECTORY_LENGTH = 20          # T = average life expectancy after retirement
TRAJECTORIES_PER_POLICY = 10        # n = number of trajectories unrolled during each policy
POLICIES_IN_BUFFER = 10             # k = number of (most recent) policies represented in buffer
LOOKBACK = POLICIES_IN_BUFFER // 2  # ell = for determining batch size D = ell.n.T

MAX_MEMORY_SIZE = POLICIES_IN_BUFFER * TRAJECTORIES_PER_POLICY * AVG_TRAJECTORY_LENGTH
BATCH_SIZE = LOOKBACK * TRAJECTORIES_PER_POLICY * AVG_TRAJECTORY_LENGTH
REPLACE = False                     # draw transitions from buffer (without replacement = False)
WAIT = LOOKBACK * TRAJECTORIES_PER_POLICY       # number of trajectories to wait before starting to take grad steps
NUM_POLICY_SAMPLES = 10             # N = number of times current policy is sampled in a gradient update

GAMMA = 0.99                        # doesn't matter too much
LR = 0.01                           # about 0.01 is good?
LR_ACTOR = LR
LR_CRITIC = LR
TAU = 0.05                           # about 0.05 - 0.2 is good. [0.01 is TOO LOW] Update for target networks

BETA = 0.04
STARTING_WEALTH = 1_000_000.
MAX_WEALTH = 10_000_000.            # this clips real wealth
WITHDRAW = BETA * STARTING_WEALTH   # constant real amount withdrawn at the end of each year,
                                    # for consumption during the next
REWARD_TYPE = 'simple'              # other alternative is 'wealth'
PENALTY = 1.

# 2) file names
CHECKPOINT_FILE_NAME = 'AC_model_weights/checkpoint.pt'
GRAPH_DIRECTORY_NAME = f'AC real graphs, S+B, smooth l1 loss'


# helper functions for training

def nominal_to_real_observation(observation):
    nominal_returns = observation[:-1]
    inflation_rate = observation[-1]

    real_returns = nominal_returns - inflation_rate  # broadcast -inflation_rate to np.array of nominal returns

    return real_returns


def real_observation_to_state(real_observation, wealth):
    return np.concatenate((real_observation, [wealth]), dtype=np.float32)


def get_financial_update(observation_real, action_old, state_old):
    # extract relevant state info for update. Uses only real inputs
    asset_returns_real = observation_real[:ACTION_DIMENSION]
    wealth_old = state_old[-1]

    # update `wealth_old` with new returns from `state`, then withdraw
    wealth = wealth_old * (1 + np.dot(asset_returns_real, action_old)) - WITHDRAW

    return wealth


def calculate_reward(state, state_old, time_remaining, penalty_, reward_type):
    # `reward_type` is in ['simple', 'wealth']

    # extract end-of-period wealth
    wealth_old = state_old[-1]
    wealth = state[-1]

    if reward_type == 'simple':
        if wealth >= 0:  # withdrawal for future time period was successful
            return 1.
        else:
            return (-penalty_) * time_remaining

    if reward_type == 'wealth':
        if wealth >= 0:
            return wealth - wealth_old
        else:
            return 0


class Trainer:
    def __init__(self,
                 num_trajectories_for_training,
                 should_maximize_entropy=False):

        self.num_trajectories_for_training = num_trajectories_for_training

        # initialize simulation and agent
        self.investor = SimulatedInvestor(countries=ALLOWED_COUNTRIES,
                                          asset_proxy_list=ALLOWED_ASSETS,
                                          macro_list=ALLOWED_MACROS,
                                          should_maximize_entropy=should_maximize_entropy)

        self.agent = ActorCritic(state_dimension=STATE_DIMENSION,
                                 action_dimension=ACTION_DIMENSION,
                                 hidden1_dimension=HIDDEN1_DIMENSION,
                                 hidden2_dimension=HIDDEN2_DIMENSION,
                                 should_clip_grads=SHOULD_CLIP_GRADS,
                                 clip_value=CLIP_VALUE,
                                 tau=TAU,
                                 lr_actor=LR,
                                 lr_critic=LR,
                                 max_memory_size=MAX_MEMORY_SIZE,
                                 batch_size=BATCH_SIZE,
                                 num_policy_samples=NUM_POLICY_SAMPLES,
                                 gamma=GAMMA,
                                 activation=ACTIVATION,
                                 replace=REPLACE,
                                 distribution_type=DISTRIBUTION_TYPE,
                                 checkpoint_file_name=CHECKPOINT_FILE_NAME)

        # declare descriptive metrics for training
        self.metric_names = ['avg stock action', 'std stock action',
                             'avg bond action', 'std bond action',
                             'financial ruin', 'terminal wealth',
                             'policy value', 'Q1 loss', 'Q2 loss']

        # initialize dictionary of zeros
        self.metric_dictionary = {metric_name: np.zeros(self.num_trajectories_for_training, dtype=np.float32)
                                  for metric_name in self.metric_names}

    def train(self):

        start_time = time.time()
        print('Beginning training...')

        for i in range(self.num_trajectories_for_training):

            # initialize wealth and withdraw amount for initial state s_{-1}
            wealth = STARTING_WEALTH

            # get chain of observations
            trajectory_length = self.investor.generate_time_after_retirement()
            if trajectory_length <= 1:
                continue
            trajectory_observations = self.investor.get_trajectory(trajectory_length)
            actions = np.zeros((trajectory_length - 1, ACTION_DIMENSION))

            # I. Take actions to update replay buffer, in order SAS'R'D'
            self.agent.policy.eval()
            with torch.inference_mode():
                for t in range(trajectory_length - 1):

                    # S: extract real `state` at the beginning of period t from nominal `observation`
                    observation = trajectory_observations[t]
                    real_observation = nominal_to_real_observation(observation)
                    state = real_observation_to_state(real_observation, wealth)

                    # A: take action, in anticipation of next state
                    action_dist = self.agent.policy.forward(torch.tensor(state))
                    action = action_dist.sample().numpy()

                    # record, for plotting
                    actions[t, :] = action

                    # S': make new observation, withdraw, update wealth, and assemble into new state
                    observation_new = trajectory_observations[t + 1]
                    real_observation_new = nominal_to_real_observation(observation_new)

                    # extract and clip new wealth
                    wealth_new = get_financial_update(real_observation_new, action, state)
                    wealth_new = np.clip(wealth_new, 0., MAX_WEALTH)

                    state_new = real_observation_to_state(real_observation_new, wealth_new)

                    # R': determine reward
                    reward_new = calculate_reward(state=state_new,
                                                  state_old=state,
                                                  penalty_=PENALTY,
                                                  reward_type=REWARD_TYPE,
                                                  time_remaining=trajectory_length - t)

                    # D': update `is_done` if either wealth = 0 or trajectory is finished:
                    is_done = False
                    if wealth_new == 0 or t + 1 == trajectory_length - 1:
                        is_done = True

                    # store SAS'R'D'
                    self.agent.memory.store_transition(state, action, state_new, reward_new, is_done)

                    # update `wealth`, ending the time step
                    wealth = wealth_new

                    # if this is the last time step, record terminal wealth, and whether you went broke
                    if is_done:
                        self.metric_dictionary['terminal wealth'][i] = wealth

                        if wealth == 0:
                            self.metric_dictionary['financial ruin'][i] = 1
                            break

            # II. Every so many trajectories, take a gradient step to update policy and Qs
            if i > WAIT and (i - 1) % TRAJECTORIES_PER_POLICY == 0:
                output = self.agent.update()
                if output is not None:
                    policy_loss, Q1_loss, Q2_loss = output

                    self.metric_dictionary['policy value'][i] = -policy_loss
                    self.metric_dictionary['Q1 loss'][i] = Q1_loss
                    self.metric_dictionary['Q2 loss'][i] = Q2_loss

            # III. Update metrics for the trajectory
            avg_action = np.mean(actions, axis=0)
            std_action = np.std(actions, axis=0)

            self.metric_dictionary['avg stock action'][i] = avg_action[0]
            self.metric_dictionary['std stock action'][i] = std_action[0]
            self.metric_dictionary['avg bond action'][i] = avg_action[1]
            self.metric_dictionary['std bond action'][i] = std_action[1]

            # save checkpoint every 100 trajectories
            if (i + 1) % 100 == 0:
                # self.agent.save_checkpoint()
                diff = time.time() - start_time
                print(f'Trajectory {i + 1} checkpoint saved. Time elapsed: {diff} seconds')

        time_elapsed = time.time() - start_time
        print(f'TRAINING COMPLETE. Time elapsed: {time_elapsed} seconds.')
        print(f'Average of {time_elapsed / self.num_trajectories_for_training} seconds / trajectory')

    def save_rolling_averages(self, window=100, should_show=False, should_save=True):

        # fill out dictionary of rolling average pd.series
        rolling_avg_dictionary = {metric: None for metric in self.metric_dictionary.keys()}
        for metric in self.metric_dictionary:
            rolling_avg_dictionary[metric] = pd.Series(self.metric_dictionary[metric]).rolling(window=window,
                                                                                               min_periods=1).mean()

        # plot
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(13, 7))

        fig_title_string = f'{DISTRIBUTION_TYPE}+' \
                           f'{ACTIVATION_NAME} agent, with ' \
                           f'(lr, tau, penalty, wait, buffer)=' \
                           f'({LR}, {TAU}, {PENALTY}, {WAIT}, {MAX_MEMORY_SIZE}).png'

        fig.suptitle(fig_title_string, fontsize=16)

        # plot stocks
        axes[0, 0].set_title(f'{window}-pt rolling avg of stock action')
        axes[0, 0].plot(rolling_avg_dictionary['avg stock action'])
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        stock_upper = rolling_avg_dictionary['avg stock action'] + rolling_avg_dictionary['std stock action']
        stock_lower = rolling_avg_dictionary['avg stock action'] - rolling_avg_dictionary['std stock action']
        axes[0, 0].fill_between(stock_lower.index, stock_lower, stock_upper, color='red', alpha=0.2)

        # plot bonds
        axes[1, 0].set_title(f'{window}-pt rolling avg of bond action')
        axes[1, 0].plot(rolling_avg_dictionary['avg bond action'])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        bond_upper = rolling_avg_dictionary['avg bond action'] + rolling_avg_dictionary['std bond action']
        bond_lower = rolling_avg_dictionary['avg bond action'] - rolling_avg_dictionary['std bond action']
        axes[1, 0].fill_between(bond_lower.index, bond_lower, bond_upper, color='red', alpha=0.2)

        # plot terminal wealth
        axes[0, 1].set_title(f'{window}-pt rolling avg terminal wealth')
        axes[0, 1].plot(rolling_avg_dictionary['terminal wealth'])
        axes[0, 1].set_ylim(0, 3e6)
        axes[0, 1].grid(True, alpha=0.3)

        # plot prob of ruin
        axes[1, 1].set_title(f'{window}-pt rolling avg prob of ruin')
        axes[1, 1].plot(rolling_avg_dictionary['financial ruin'])
        axes[1, 1].set_ylim(0, 0.25)
        axes[1, 1].grid(True, alpha=0.3)

        # plot policy loss
        axes[0, 2].set_title(f'{window}-pt rolling avg policy value')
        axes[0, 2].plot(rolling_avg_dictionary['policy value'])
        #axes[0, 2].set_ylim(-10, 10)
        axes[0, 2].grid(True, alpha=0.3)

        # plot Q1, Q2 losses together on last
        axes[1, 2].set_title(f'{window}-pt rolling avg Q1, Q2 MSE losses')
        axes[1, 2].plot(rolling_avg_dictionary['Q1 loss'])
        rolling_avg_dictionary['Q2 loss'].plot(ax=axes[1, 2], color='red', linestyle='--')
        #axes[1, 2].set_ylim(0, 20)
        axes[1, 2].grid(True, alpha=0.3)

        fig.tight_layout()

        if should_save:
            os.makedirs(GRAPH_DIRECTORY_NAME, exist_ok=True)
            file_path = os.path.join(GRAPH_DIRECTORY_NAME, fig_title_string)
            print(f'the filepath title is {file_path}')

            plt.savefig(file_path)

        if should_show:
            plt.show()


### CALL FUNCTIONS HERE ###

# 2) hyperparameters to play around with (those listed here are defaults)
NUM_TRAJECTORIES_FOR_TRAINING = 1000  # 1000-1500 seems good; training seems to stabilize

lrs = [0.01]
taus = [0.1, 0.2]
penalties = [1, 5, 10]

for lr in lrs:
    for tau in taus:
        for penalty in penalties:
            LR = lr
            TAU = tau
            PENALTY = penalty
            print(f'global variables reset! {LR, TAU, PENALTY}')
            trainer = Trainer(num_trajectories_for_training=NUM_TRAJECTORIES_FOR_TRAINING)
            trainer.train()
            trainer.save_rolling_averages(should_save=True, should_show=False)

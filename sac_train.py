"""
    Version: Thursday, March 12, 2026
"""

import torch
from torch import nn
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

ALLOWED_ASSETS = [['eq_tr', 'eq_capgain', 'eq_div_rtn'],
                  ['bond_tr', 'bond_rate']]  # just keep two

ALLOWED_MACROS = [['cpi']]  # always keep cpi last (for inflation calculation)

ACTION_DIMENSION = len(ALLOWED_ASSETS)
OBSERVATION_DIMENSION = len(ALLOWED_ASSETS + ALLOWED_MACROS) - 1
STATE_DIMENSION = OBSERVATION_DIMENSION + 1     # for adding wealth.

"""
IMPORTANT:
    `observations` are NOMINAL return vectors, including inflation (return on cpi)
    observation[-1] = inflation rate, by convention
    
    `states` are REAL return vectors:
    state = observation_real + wealth_real

    SO:
    state[-1] = real wealth
    observation[:ACTION_DIMENSION] = real asset rtns
    
    To convert an observation to a state, first produce a `real observation` = observation[:-2] - observation[-1]
"""

# III. Set up agent and simulation

# 1) FIXED hyperparameters
HIDDEN1_DIMENSION = 1
HIDDEN2_DIMENSION = 1

BATCH_SIZE = 20  # make it big enough to lower variance, but not so big that it's maximal
NUM_TRAJECTORIES = 3_000  # 1500-2000 seems good; training seems to stabilize
REPLACE = True  # draw with replacement?
GAMMA = 0.99
LR = 0.01  # good if close to 0.01
LR_ACTOR = LR
LR_CRITIC = LR
LR_ALPHA = LR  # Temperature learning rate
TAU = 0.1  # update for the two Q_target networks. Good if between 0.1 and 0.2

BETA = 0.04
STARTING_WEALTH = 1_000_000.
MAX_WEALTH = 10_000_000.                # will clip real wealth!
WITHDRAW = BETA * STARTING_WEALTH

H_TARGET = -ACTION_DIMENSION
ALPHA_INITIAL = 1.

activations = {'relu': nn.ReLU(inplace=True),
               'tanh': nn.Tanh()}

# 2) hyperparameters to play around with (those listed here will be the defaults for the trainer class)
PENALTY = 1.
MAX_MEMORY_SIZE = 3_000  # about 1x-2x the number 3000 of unique state-state transitions
WAIT = 20  # how many trajectories to wait before starting gradient steps (say 50% of dataset)
DISTRIBUTION_TYPE = 'gaussian'  # 'gaussian' is better than 'dirichlet'
ACTIVATION_NAME = 'relu'  # 'relu' is better than 'tanh'


# helper functions for training

def nominal_to_real_observation(observation):

    nominal_returns = observation[:-1]
    inflation_rate = observation[-1]

    real_returns = nominal_returns - inflation_rate     # broadcast -inflation_rate to np.array of nominal returns

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

    def __init__(self, reward_type='simple',
                 gamma=GAMMA,
                 penalty=PENALTY,
                 distribution_type=DISTRIBUTION_TYPE,
                 activation_name=ACTIVATION_NAME,
                 max_memory_size=MAX_MEMORY_SIZE,
                 wait=WAIT,
                 lr=LR,
                 tau=TAU,
                 should_maximize_entropy=False):

        # for titling plots
        self.reward_type = reward_type
        self.gamma = gamma
        self.penalty = penalty
        self.max_memory_size = max_memory_size
        self.distribution_type = DISTRIBUTION_TYPE
        self.activation_name = activation_name  # string
        self.activation = activations[self.activation_name]  # actual nn. object
        self.lr = lr
        self.tau = tau
        self.wait = wait

        # initialize simulation and agent
        self.investor = SimulatedInvestor(countries=ALLOWED_COUNTRIES,
                                          asset_proxy_list=ALLOWED_ASSETS,
                                          macro_list=ALLOWED_MACROS,
                                          should_maximize_entropy=should_maximize_entropy)

        self.agent = SoftActorCritic(state_dimension=STATE_DIMENSION,
                                     action_dimension=ACTION_DIMENSION,
                                     hidden1_dimension=HIDDEN1_DIMENSION,
                                     hidden2_dimension=HIDDEN2_DIMENSION,
                                     tau=self.tau,
                                     lr_actor=self.lr,
                                     lr_critic=self.lr,
                                     lr_alpha=self.lr,
                                     H_target=H_TARGET,
                                     max_memory_size=self.max_memory_size,
                                     batch_size=BATCH_SIZE,
                                     alpha_initial=ALPHA_INITIAL,
                                     gamma=self.gamma,
                                     activation=self.activation,
                                     replace=REPLACE,
                                     distribution_type=distribution_type,
                                     checkpoint_file_name="SAC_model_weights/checkpoint.pt")

        # declare descriptive metrics for training
        self.metric_names = ['avg stock action', 'std stock action',
                             'avg bond action', 'std bond action',
                             'financial ruin', 'terminal wealth',
                             'policy loss', 'Q1 loss', 'Q2 loss']

        # initialize dictionary of zeros
        self.metric_dictionary = {metric_name: np.zeros(NUM_TRAJECTORIES, dtype=np.float32)
                                  for metric_name in self.metric_names}

    def train(self):

        start_time = time.time()
        print('Beginning training...')

        for i in range(NUM_TRAJECTORIES):

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
                                                  reward_type=self.reward_type,
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

            # II. Take a gradient step, after having waited for buffer to fill
            if i > self.wait:
                policy_loss, Q1_loss, Q2_loss = self.agent.update()
                self.metric_dictionary['policy loss'][i] = policy_loss

                # declare reasonable cutoffs for Q-values
                comparison = 0
                if self.reward_type == 'wealth':
                    comparison = 10 * MAX_WEALTH
                elif self.reward_type == 'simple':
                    comparison = 20

                self.metric_dictionary['Q1 loss'][i] = min(Q1_loss, comparison)
                self.metric_dictionary['Q2 loss'][i] = min(Q2_loss, comparison)

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
        print(f'Average of {time_elapsed / NUM_TRAJECTORIES} seconds / trajectory')

    def save_rolling_averages(self, window=100, should_show=False, should_save=True):

        # fill out dictionary of rolling average pd.series
        rolling_avg_dictionary = {metric: None for metric in self.metric_dictionary.keys()}
        for metric in self.metric_dictionary:
            rolling_avg_dictionary[metric] = pd.Series(self.metric_dictionary[metric]).rolling(window=window,
                                                                                               min_periods=1).mean()

        # plot
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(13, 7))

        fig_title_string = f'(S+B, {self.reward_type} rewards) {self.distribution_type}+' \
                           f'{self.activation_name} agent, with ' \
                           f'(wait, penalty, buffer)=' \
                           f'({self.wait}, {self.penalty}, {self.max_memory_size})'

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
        axes[0, 2].set_title(f'{window}-pt rolling avg policy loss')
        axes[0, 2].plot(rolling_avg_dictionary['policy loss'])
        axes[0, 2].grid(True, alpha=0.3)

        # plot Q1, Q2 losses together on last
        axes[1, 2].set_title(f'{window}-pt rolling avg Q1, Q2 losses')
        axes[1, 2].plot(rolling_avg_dictionary['Q1 loss'])
        rolling_avg_dictionary['Q2 loss'].plot(ax=axes[1, 2], color='red', linestyle='--')
        axes[1, 2].grid(True, alpha=0.3)

        fig.tight_layout()

        if should_save:
            base_path = '/Users/danielchupin/PycharmProjects/InvestingBot/agents with different buffers/'
            file_path = base_path + fig_title_string + '.png'
            plt.savefig(file_path)

        if should_show:
            plt.show()


### CALL FUNCTIONS HERE ###

max_memory_sizes = [1000]
waits = [20]
taus = [0.15]
penalties = [5, 10]
reward_types = ['simple']
should_maximize_entropy = [True, False]

for penalty in penalties:
    for reward_type in reward_types:
        for b in should_maximize_entropy:
            trainer = Trainer(wait=20,
                              tau=0.15,
                              max_memory_size=1000,
                              reward_type=reward_type,
                              penalty=penalty,
                              should_maximize_entropy=b)
            trainer.train()
            trainer.save_rolling_averages(should_save=False, should_show=True)

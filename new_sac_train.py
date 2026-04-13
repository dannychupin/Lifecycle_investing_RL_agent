"""
    Version: April 10, 2026

"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
import math

from block_bootstrap import SimulatedInvestor
from new_sac_agent import SoftActorCritic

# seeds for debugging
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

#####################################
# I. DETERMINE COUNTRIES AND ASSETS #
#####################################

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

# states are [real returns on assets] + [other real macro returns] + [wealth]
STATE_DIMENSION = len(ALLOWED_ASSETS) + (len(ALLOWED_MACROS) - 1) + 1       # -1 to remove inflation

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

#######################
# II. HYPERPARAMETERS #
#######################


def volume_of_simplex(n):
    # log(sqrt(n+1)/n!)
    return 0.5 * math.log(n + 1) - math.lgamma(n + 1)


def get_beta(i, n):
    # linearly taper from 1 at i=0 to 0 at i=n
    return max(0.0, 1 - i / n)


# 1) default hyperparameters (DO NOT CHANGE OR ERASE)
HIDDEN1_DIMENSION = 8
HIDDEN2_DIMENSION = 8
VOLUME_FRACTION = 1 / 5     # volume fraction of n-simplex that we want to encourage policy to occupy
ENTROPY_MIN = math.log(VOLUME_FRACTION) + volume_of_simplex(ACTION_DIMENSION)   # -1.8 for n=2
BETA_MAX = 0.0

DISTRIBUTIONS = ['gaussian', 'dirichlet']
DISTRIBUTION_NAME = 'gaussian'              # 'gaussian' seems better than 'dirichlet'

ACTIVATIONS = {'relu': nn.ReLU(),
               'tanh': nn.Tanh()}
ACTIVATION_NAME = 'tanh'                    # 'tanh' seems better than 'relu'
ACTIVATION = ACTIVATIONS[ACTIVATION_NAME]

SHOULD_CLIP_GRADS = True
CLIP_VALUE = 10.0

# annotated with notation from LaTeX document
AVG_TRAJECTORY_LENGTH = 20                  # T = average life expectancy after retirement
TRAJECTORIES_PER_POLICY = 10                 # n = number of trajectories unrolled during each policy
POLICIES_IN_BUFFER = 10                     # k = number of (most recent) policies represented in buffer
LOOKBACK = POLICIES_IN_BUFFER // 10          # ell = for determining batch size D = ell.n.T

MAX_MEMORY_SIZE = POLICIES_IN_BUFFER * TRAJECTORIES_PER_POLICY * AVG_TRAJECTORY_LENGTH
BATCH_SIZE = LOOKBACK * TRAJECTORIES_PER_POLICY * AVG_TRAJECTORY_LENGTH
REPLACE = False                             # draw transitions from buffer (without replacement = False)
WAIT = POLICIES_IN_BUFFER * TRAJECTORIES_PER_POLICY   # number of trajectories to wait before starting to take grad steps
NUM_POLICY_SAMPLES = 1                     # N = number of times current policy is sampled in a gradient update

GAMMA = 0.99                                # doesn't matter too much
LR = 0.0003                                  #
LR_ACTOR = 2 * LR
LR_CRITIC = LR
TAU = 0.0005                                  # slow it down below 0.005

BETA = 0.04
STARTING_WEALTH = 1_000_000.
MAX_WEALTH = 10_000_000.                    # this clips real wealth
WITHDRAW = BETA * STARTING_WEALTH           # constant real amount withdrawn at the end of each year,
                                            # for consumption during the next
REWARD_TYPE = 'simple'                      # 'simple' and 'wealth'
PENALTY = 10.                                # only relevant for 'simple'. 1 is a good default

# 2) file names
CHECKPOINT_FILE_NAME = 'SAC_model_weights/checkpoint.pt'
GRAPH_DIRECTORY_NAME = f'SAC {REWARD_TYPE} synthetic data gaussians with rho=1, dist and act comparison'


#################
# III. TRAINING #
#################

def nominal_to_real_observation(observation):
    nominal_returns = observation[:-1]
    inflation_rate = observation[-1]

    real_returns = nominal_returns - inflation_rate  # broadcast -inflation_rate to np.array of nominal returns

    return real_returns


def real_observation_to_state(real_observation, wealth):
    return np.concatenate((real_observation, [wealth]), dtype=np.float32)


def earn_returns_and_make_withdrawal(observation_real, action_old, wealth_old):
    # extract relevant state info for update. Uses only REAL INPUTS!
    asset_returns_real = observation_real[:ACTION_DIMENSION]

    # update `wealth_old` with new returns from `state`, then withdraw
    wealth_new = wealth_old * (1 + np.dot(asset_returns_real, action_old)) - WITHDRAW

    return wealth_new


def calculate_reward(wealth_new, wealth_old, penalty_, reward_type):
    if reward_type == 'simple':
        if wealth_new > 0:
            reward = 1 + math.log(wealth_new/WITHDRAW + 1)
            return reward
        else:
            return -penalty_

    if reward_type == 'wealth':
        if wealth_new > 0:
            return wealth_new - wealth_old
        else:
            return -1e6


class Trainer:
    def __init__(self,
                 num_trajectories_for_training,
                 synthetic_mode,
                 activation_name,
                 distribution_name,
                 is_synthetic,
                 near_ruin_cutoff,
                 graph_on,
                 is_entropy_maximizing=False):

        self.num_trajectories_for_training = num_trajectories_for_training
        self.activation_name = activation_name
        self.distribution_name = distribution_name
        self.is_synthetic = is_synthetic
        self.synthetic_mode = synthetic_mode
        self.near_ruin_cutoff = near_ruin_cutoff
        self.graph_on = graph_on

        # initialize simulation and agent
        self.investor = SimulatedInvestor(countries=ALLOWED_COUNTRIES,
                                          asset_proxy_list=ALLOWED_ASSETS,
                                          macro_list=ALLOWED_MACROS,
                                          is_entropy_maximizing=is_entropy_maximizing)

        self.agent = SoftActorCritic(state_dimension=STATE_DIMENSION,
                                     action_dimension=ACTION_DIMENSION,
                                     hidden1_dimension=HIDDEN1_DIMENSION,
                                     hidden2_dimension=HIDDEN2_DIMENSION,
                                     should_clip_grads=SHOULD_CLIP_GRADS,
                                     clip_value=CLIP_VALUE,
                                     tau=TAU,
                                     lr_actor=LR_ACTOR,
                                     lr_critic=LR_CRITIC,
                                     entropy_min=ENTROPY_MIN,
                                     max_memory_size=MAX_MEMORY_SIZE,
                                     batch_size=BATCH_SIZE,
                                     num_policy_samples=NUM_POLICY_SAMPLES,
                                     gamma=GAMMA,
                                     activation=ACTIVATIONS[self.activation_name],
                                     replace=REPLACE,
                                     distribution_type=self.distribution_name,
                                     checkpoint_file_name=CHECKPOINT_FILE_NAME)

        # declare descriptive metrics for training
        self.metric_names = ['avg stock action', 'std stock action',
                             'avg bond action', 'std bond action',
                             'financial ruin', 'terminal wealth']

        self.objective_names = ['policy value',
                                'expected min of Qs',
                                'expected entropy',
                                'Q1 loss',
                                'Q2 loss']

        # initialize dictionaries of zeros
        self.metric_dictionary = {metric_name: np.zeros(self.num_trajectories_for_training, dtype=np.float32)
                                  for metric_name in self.metric_names}

        num_objective_entries = self.num_trajectories_for_training // TRAJECTORIES_PER_POLICY
        self.objective_dictionary = {objective_name: np.zeros(num_objective_entries, dtype=np.float32)
                                     for objective_name in self.objective_names}

    def train(self):

        start_time = time.time()
        print('Beginning training...')

        for i in range(self.num_trajectories_for_training):
            # visualize distributions
            if self.agent.normal_memory.memory_counter >= self.agent.normal_memory.batch_size:
                if self.agent.ruin_memory.memory_counter >= self.agent.ruin_memory.batch_size:
                    if i in self.graph_on:
                        self.agent.visualize_nr_distributions(num_samples=100, traj_id=i)

            # initialize wealth and withdraw amount for initial state s_{-1}
            wealth = STARTING_WEALTH

            # get chain of observations
            trajectory_length = self.investor.generate_time_after_retirement()
            if trajectory_length <= 1:
                continue

            trajectory_observations = None
            if self.is_synthetic:
                trajectory_observations = self.investor.get_synthetic_trajectory(trajectory_length,
                                                                                 synthetic_mode=self.synthetic_mode)
            else:
                trajectory_observations = self.investor.get_trajectory(trajectory_length)

            # for calculating avg action later, to track progress
            actions = []

            # I. Take actions to update replay buffer, in order SAS'R'D'
            with torch.inference_mode():
                for t in range(trajectory_length - 1):

                    # S: extract real `state` at the beginning of period t from nominal `observation`
                    observation = trajectory_observations[t]

                    real_observation = None
                    if self.is_synthetic:
                        real_observation = observation
                    else:
                        real_observation = nominal_to_real_observation(observation)

                    state = real_observation_to_state(real_observation, wealth)

                    # A: take action, in anticipation of next state
                    action_dist = self.agent.policy.forward(torch.from_numpy(state))
                    action = action_dist.sample().numpy()

                    # start of S': make new observation, withdraw, update new wealth
                    observation_new = trajectory_observations[t + 1]

                    real_observation_new = None
                    if self.is_synthetic:
                        real_observation_new = observation_new
                    else:
                        real_observation_new = nominal_to_real_observation(observation_new)

                    wealth_new_raw = earn_returns_and_make_withdrawal(real_observation_new, action, wealth)

                    # R': determine reward
                    reward_new = calculate_reward(wealth_new=wealth_new_raw,
                                                  wealth_old=wealth,
                                                  penalty_=PENALTY,
                                                  reward_type=REWARD_TYPE)

                    # end of S': clip wealth, and create new state
                    wealth_new = np.clip(wealth_new_raw, a_min=0.0, a_max=MAX_WEALTH)
                    state_new = real_observation_to_state(real_observation_new, wealth_new)

                    # D': update `is_done`
                    is_done = False

                    # record action, for plotting
                    if is_done is False:
                        actions.append(action)

                    if wealth_new <= 0.0 or t + 1 == trajectory_length - 1:
                        is_done = True

                    # store SAS'R'D'
                    if wealth_new / WITHDRAW < self.near_ruin_cutoff:
                        self.agent.ruin_memory.store_transition(state, action, state_new, reward_new, is_done)
                    else:
                        self.agent.normal_memory.store_transition(state, action, state_new, reward_new, is_done)

                    # update `wealth`, ending the time step
                    wealth = wealth_new  # == 0 iff experienced ruin

                    # if this is the last time step, record terminal wealth, and whether you went broke
                    if is_done:
                        self.metric_dictionary['terminal wealth'][i] = wealth

                        if wealth <= 0.0:
                            self.metric_dictionary['financial ruin'][i] = 1
                            break

            # II. Every so many trajectories, take a gradient step to update policy and Qs
            if i > WAIT and (i - 1) % TRAJECTORIES_PER_POLICY == 0:

                cutoff = min(1000, NUM_TRAJECTORIES_FOR_TRAINING // 2)
                beta = BETA_MAX * get_beta(i=i, n=cutoff)

                print(f'## Trajectory {i-1} update ##')
                output = self.agent.update(beta=beta)

                if output is not None:
                    policy_value, expected_min_of_Qs, expected_entropy, Q1_loss, Q2_loss = output

                    dictionary_index = i // TRAJECTORIES_PER_POLICY
                    self.objective_dictionary['policy value'][dictionary_index] = policy_value
                    self.objective_dictionary['expected min of Qs'][dictionary_index] = expected_min_of_Qs
                    self.objective_dictionary['expected entropy'][dictionary_index] = expected_entropy
                    self.objective_dictionary['Q1 loss'][dictionary_index] = Q1_loss
                    self.objective_dictionary['Q2 loss'][dictionary_index] = Q2_loss

            # III. Update metrics for the trajectory
            actions = np.vstack(actions)
            avg_action = np.mean(actions, axis=0)

            """for troubleshooting:
            sum = avg_action[0] + avg_action[1]
            if sum != 1.0:
                print(f'UH OH!!! action sum is {sum}')
            """

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

    def get_rolling_averages(self, window=100, should_show=False, should_save=True):

        # fill out dictionaries of rolling average pd.series
        rolling_avg_metric_dictionary = {metric: None for metric in self.metric_dictionary.keys()}
        for metric in self.metric_dictionary:
            rolling_avg_metric_dictionary[metric] = \
                pd.Series(self.metric_dictionary[metric]).rolling(window=window, min_periods=1).mean()
        last_stock_action = rolling_avg_metric_dictionary['avg stock action'].iloc[-1]
        last_terminal_wealth = rolling_avg_metric_dictionary['terminal wealth'].iloc[-1]

        rolling_avg_objective_dictionary = {objective: None for objective in self.objective_dictionary.keys()}
        objective_window = window // TRAJECTORIES_PER_POLICY
        for objective in self.objective_dictionary:
            rolling_avg_objective_dictionary[objective] = \
                pd.Series(self.objective_dictionary[objective]).rolling(window=objective_window, min_periods=1).mean()

        last_policy_value = rolling_avg_objective_dictionary['policy value'].iloc[-1]
        last_Q1_MSE_loss = rolling_avg_objective_dictionary['Q1 loss'].iloc[-1]

        # plot
        num_gradient_updates = self.num_trajectories_for_training // TRAJECTORIES_PER_POLICY

        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, figsize=(13, 7))

        fig_title_string = f'{self.distribution_name}+' \
                           f'{self.activation_name} agent, with ' \
                           f'(lr, tau, penalty, wait, buffer)=' \
                           f'({LR}, {TAU}, {PENALTY}, {WAIT}, {MAX_MEMORY_SIZE}).png'

        fig.suptitle(fig_title_string, fontsize=16)

        # plot stocks
        axes[0, 0].set_title(f'{window}-pt rolling avg of stock action')
        axes[0, 0].plot(rolling_avg_metric_dictionary['avg stock action'])
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        stock_upper = rolling_avg_metric_dictionary['avg stock action'] + rolling_avg_metric_dictionary[
            'std stock action']
        stock_lower = rolling_avg_metric_dictionary['avg stock action'] - rolling_avg_metric_dictionary[
            'std stock action']
        axes[0, 0].fill_between(stock_lower.index, stock_lower, stock_upper, color='red', alpha=0.2)

        # plot bonds
        axes[1, 0].set_title(f'{window}-pt rolling avg of bond action')
        axes[1, 0].plot(rolling_avg_metric_dictionary['avg bond action'])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        bond_upper = rolling_avg_metric_dictionary['avg bond action'] + rolling_avg_metric_dictionary['std bond action']
        bond_lower = rolling_avg_metric_dictionary['avg bond action'] - rolling_avg_metric_dictionary['std bond action']
        axes[1, 0].fill_between(bond_lower.index, bond_lower, bond_upper, color='red', alpha=0.2)

        # plot terminal wealth
        axes[0, 1].set_title(f'{window}-pt rolling avg terminal wealth')
        axes[0, 1].plot(rolling_avg_metric_dictionary['terminal wealth'])
        axes[0, 1].set_ylim(0, 3e6)
        axes[0, 1].grid(True, alpha=0.3)

        # plot prob of ruin
        axes[1, 1].set_title(f'{window}-pt rolling avg prob of ruin')
        axes[1, 1].plot(rolling_avg_metric_dictionary['financial ruin'])
        axes[1, 1].set_ylim(0, 0.25)
        axes[1, 1].grid(True, alpha=0.3)

        # plot policy loss
        axes[0, 2].set_title(f'{objective_window}-pt rolling avg policy value')
        axes[0, 2].plot(rolling_avg_objective_dictionary['policy value'])
        rolling_avg_objective_dictionary['expected min of Qs'].plot(ax=axes[0, 2], color='red', linestyle='--')
        rolling_avg_objective_dictionary['expected entropy'].plot(ax=axes[0, 2], color='orange', linestyle='--')
        axes[0, 2].set_xlim(0, num_gradient_updates)
        axes[0, 2].grid(True, alpha=0.3)

        # plot Q1, Q2 losses together on last
        axes[1, 2].set_title(f'{objective_window}-pt rolling avg Q1, Q2 MSE losses')
        axes[1, 2].plot(rolling_avg_objective_dictionary['Q1 loss'])
        rolling_avg_objective_dictionary['Q2 loss'].plot(ax=axes[1, 2], color='red', linestyle='--')
        axes[1, 2].set_xlim(0, num_gradient_updates)
        axes[1, 2].grid(True, alpha=0.3)

        fig.tight_layout()

        if should_save:
            os.makedirs(GRAPH_DIRECTORY_NAME, exist_ok=True)
            file_path = os.path.join(GRAPH_DIRECTORY_NAME, fig_title_string)

            plt.savefig(file_path)

        if should_show:
            plt.show()

        return last_stock_action, last_terminal_wealth, last_policy_value, last_Q1_MSE_loss


#######################
# CALL FUNCTIONS HERE #
#######################

NUM_TRAJECTORIES_FOR_TRAINING = 20_000
GRAPH_ON = []
IS_SYNTHETIC = True
SYNTHETIC_MODE = 'correlated gaussians'
NEAR_RUIN_CUTOFF = 2.

trainer = Trainer(is_synthetic=IS_SYNTHETIC,
                  synthetic_mode=SYNTHETIC_MODE,
                  activation_name='tanh',
                  distribution_name='dirichlet',
                  near_ruin_cutoff=NEAR_RUIN_CUTOFF,
                  graph_on=GRAPH_ON,
                  num_trajectories_for_training=NUM_TRAJECTORIES_FOR_TRAINING)
trainer.train()
_, _, _, _ = trainer.get_rolling_averages(should_save=False, should_show=True)


def compare_dists_and_activations(num_seeds, bins=10):
    for distribution_name in DISTRIBUTIONS:
        for activation_name in ACTIVATIONS.keys():
            print(f'entering dist={distribution_name} and act={activation_name}')

            last_data = np.zeros(shape=(num_seeds, 4))

            # set new seeds and train
            for seed in range(num_seeds):
                print(f'entering seed={seed}')
                SEED = seed
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                trainer = Trainer(num_trajectories_for_training=NUM_TRAJECTORIES_FOR_TRAINING,
                                  activation_name=activation_name,
                                  distribution_name=distribution_name,
                                  is_synthetic=IS_SYNTHETIC,
                                  near_ruin_cutoff=NEAR_RUIN_CUTOFF,
                                  synthetic_mode=SYNTHETIC_MODE)
                trainer.train()
                last_data[seed, :] = trainer.get_rolling_averages(should_save=False, should_show=False)

            # plot
            title_string = f'{distribution_name}+{activation_name}, for {NUM_TRAJECTORIES_FOR_TRAINING} trajectories.png'
            fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False, figsize=(13, 7))
            axes = axes.ravel()
            fig.suptitle(title_string, fontsize=16)

            axes[0].set_title('final learned stock action')
            axes[0].hist(last_data[:, 0], range=(0, 1), bins=bins)

            axes[1].set_title('final terminal wealth')
            axes[1].hist(last_data[:, 1], range=(0, MAX_WEALTH), bins=bins)

            axes[2].set_title('final policy value')
            axes[2].hist(last_data[:, 2], bins=bins)

            axes[3].set_title('final Q1 MSE loss')
            axes[3].hist(last_data[:, 3], bins=bins)

            os.makedirs(GRAPH_DIRECTORY_NAME, exist_ok=True)
            file_name = os.path.join(GRAPH_DIRECTORY_NAME, title_string)
            plt.savefig(file_name)


#compare_dists_and_activations(20)

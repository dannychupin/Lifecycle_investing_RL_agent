import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import floor
import time

from block_bootstrap import SimulatedInvestor

### 0. BACKGROUND/REFERENCE. ###
# Don't edit this section; change values outside it.

# set seed for debugging
SEED = 42
np.random.seed(SEED)

# change these in ablations (e.g. post-WWII changes things?)
MIN_YEAR = 1871     # set to 1870+1 to calculate %change from prev year of macro variables (cpi, gdp, wages)
MAX_YEAR = 2020
AVG_RESIDENCE_LENGTH = 10  # take as the mean of geometric distribution

# change these in ablations (to shape what kind of 'average investor' we are modeling)
COUNTRIES = ['Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
             'Denmark', 'Spain', 'Finland', 'France', 'UK',
             'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
             'Portugal', 'Sweden', 'USA']   # 18 countries

# hyperparameters for simulation
NUM_TRAJECTORIES = 200
STARTING_WEALTH = 1_000_000
BETA = 0.04


### I. SIMULATION ###

def get_financial_update(observation, weights, wealth_old, withdraw_old):

    # extract info
    asset_returns = observation[:ACTION_DIMENSION]
    inflation_rate = observation[-1]

    # update `withdraw_old` to keep up with recent inflation
    withdraw = (1 + inflation_rate) * withdraw_old

    # update `wealth_old` with new returns from `state`, then withdraw
    wealth = wealth_old * (1 + np.dot(asset_returns, weights)) - withdraw

    return wealth, withdraw


def try_strategies(avg_length_of_residence,
                   country_probabilities,
                   strategies,
                   allowed_countries=COUNTRIES,
                   num_trajectories=NUM_TRAJECTORIES):

    # initialize simulation
    investor = SimulatedInvestor(countries=allowed_countries,
                                 asset_proxy_list=ALLOWED_ASSETS,
                                 macro_list=ALLOWED_MACROS,
                                 country_probabilities=country_probabilities,
                                 avg_residence_length=avg_length_of_residence)

    num_strategies = len(strategies)

    terminal_wealth = np.zeros(shape=(num_trajectories, num_strategies))
    financial_ruin = np.zeros(shape=(num_trajectories, num_strategies))

    start_time = time.time()
    print('Beginning simulation...')

    for i in range(num_trajectories):

        # initialize wealth, and get trajectory
        wealth_list = np.full(shape=num_strategies, fill_value=STARTING_WEALTH)
        withdraw_list = np.full(shape=num_strategies, fill_value=BETA*STARTING_WEALTH)
        trajectory_length = investor.generate_time_after_retirement()
        trajectory = investor.get_trajectory(trajectory_length)

        for t in range(trajectory_length):

            # get observation
            observation = trajectory[t]

            # withdraw, and update wealth
            for j, weights in enumerate(strategies):

                wealth = wealth_list[j]
                withdraw = withdraw_list[j]

                # get updates
                wealth_new, withdraw_new = get_financial_update(observation=observation,
                                                                weights=weights,
                                                                wealth_old=wealth,
                                                                withdraw_old=withdraw)

                # check to see if I'm broke; if so,
                if wealth_new <= 0:
                    wealth_new = 0
                    withdraw_new = 0
                    financial_ruin[i, j] = 1

                # update values
                wealth_list[j] = wealth_new
                withdraw_list[j] = withdraw_new

            # at the end of trajectory, record terminal wealth
            if t == trajectory_length - 1:
                terminal_wealth[i] = wealth_list

        if (i + 1) % 100 == 0:
            print(f'Trajectory {i + 1} complete!')

    time_elapsed = time.time() - start_time
    print(f'SIMULATION COMPLETE. Time elapsed: {time_elapsed} seconds.')

    financial_ruin_list = (1/num_trajectories) * financial_ruin.sum(axis=0)

    return terminal_wealth, financial_ruin_list

# optionally truncate distributions at this wealth
MAX_TERMINAL_WEALTH = 1e7       # serves as max wealth window in histograms


def plot_and_save_results(terminal_wealth,
                          financial_ruin_list,
                          strategies,
                          avg_residence_len,
                          probs_name,
                          save_filename,
                          num_trajectories=NUM_TRAJECTORIES,
                          should_truncate=True,
                          num_bins=100):

    # optionally truncate. This is to give sensible means to mostly-stock strategies
    if should_truncate:
        terminal_wealth[terminal_wealth > MAX_TERMINAL_WEALTH] = MAX_TERMINAL_WEALTH

    fig, axes = plt.subplots(nrows=2, ncols=len(strategies)//2+1, figsize=(14, 7))
    axes = axes.ravel()

    range_hist = (0, MAX_TERMINAL_WEALTH)
    y_lim = (1/15) * NUM_TRAJECTORIES

    strategy_names = [f'{int(strategy[0] * 100)}% stocks' for strategy in strategies]

    # first plots are terminal wealths for strategies

    for i, strategy_name in enumerate(strategy_names):

        # get terminal wealth
        terminal_wealth_i = terminal_wealth[:, i]

        # plot
        axes[i].hist(np.array(terminal_wealth_i), bins=num_bins, range=range_hist)
        axes[i].set_title(strategy_name)
        axes[i].set_xlabel('Wealth')
        axes[i].set_ylim(0, y_lim)

    # second-to-last plot is means and stdevs
    means = np.mean(terminal_wealth, axis=0)
    stdevs = np.std(terminal_wealth, axis=0)

    axes[-2].errorbar(strategy_names, means, yerr=stdevs, fmt='-o', ecolor='red', capsize=5)
    axes[-2].set_title('Mean and Stdev')
    axes[-2].set_xlabel('Category')

    # last plot is probability of ruin
    axes[-1].bar(np.array(strategy_names), financial_ruin_list)
    axes[-1].set_title(f'Prob(financial ruin)')

    fig.suptitle(f'Terminal wealth for given % stocks ({avg_residence_len} yr avg len, '
                 f'{probs_name}, {num_trajectories} trajectories)', fontsize=16)
    fig.tight_layout()
    plt.savefig(save_filename)
    # plt.show()


### III. RUN ON CONSTANT ALLOCATION STRATEGIES ###

"""Here we will sample a bunch of regimes
    1. uniform vs market cap-weighted country selections
    2. gradation of strategies between 100% stock and 100% bond
    3. avg_residence_length in {1, 5, 10}
"""


def main(num_strategies,
         avg_len_list,
         probabilities_dict,
         num_trajectories):

    # create array of strategies
    strategies = [[1.0 - i / (num_strategies - 1), 0.0 + i / (num_strategies - 1)] for i in range(num_strategies)]
    strategies = np.array(strategies)

    base_path = '/Users/danielchupin/PycharmProjects/InvestingBot/graphs/'

    for len in avg_len_list:
        for probs_name in probabilities_dict.keys():
            terminal_wealth, financial_ruin_list = try_strategies(avg_length_of_residence=len,
                                                                  strategies=strategies,
                                                                  num_trajectories=num_trajectories,
                                                                  country_probabilities=probabilities_dict[probs_name],
                                                                  allowed_countries=ALLOWED_COUNTRIES)

            file_name = f'Term wealth ({len} yrs, {probs_name}, {num_trajectories} traj).png'
            complete_file_path = base_path + file_name

            plot_and_save_results(terminal_wealth=terminal_wealth,
                                  financial_ruin_list=financial_ruin_list,
                                  strategies=strategies,
                                  avg_residence_len=len,
                                  probs_name=probs_name,
                                  save_filename=complete_file_path,
                                  num_trajectories=num_trajectories,
                                  should_truncate=True,
                                  num_bins=100)

    print('All done!')

ALLOWED_COUNTRIES = ['Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
                     'Denmark', 'Spain', 'Finland', 'France', 'UK',
                     'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
                     'Portugal', 'Sweden', 'USA']  # keep all

MARKET_CAPS = {"Australia": 1700, "Belgium": 250, "Canada": 4035, "Switzerland": 3089, "Germany": 3112,
               "Denmark": 700, "Spain": 900, "Finland": 350, "France": 3376, "UK": 4552,
               "Ireland": 300, "Italy": 800, "Japan": 7165, "Netherlands": 1500, "Norway": 450,
               "Portugal": 110, "Sweden": 1100, "USA": 70384}       # in billions USD, 2025-2026 estimate

MARKET_CAPS_LIST = np.array(list(MARKET_CAPS.values()))
MARKET_CAPS_PROBABILITIES = MARKET_CAPS_LIST / MARKET_CAPS_LIST.sum()
UNIFORM_PROBABILITIES = np.array([1/len(ALLOWED_COUNTRIES) for _ in ALLOWED_COUNTRIES])

probabilities_dict = {'uniform wts': UNIFORM_PROBABILITIES, 'mkt cap wts': MARKET_CAPS_PROBABILITIES}

ALLOWED_ASSETS = [['eq_tr', 'eq_capgain', 'eq_div_rtn'], ['bond_tr', 'bond_rate']]  # just keep equity and bonds
ALLOWED_MACROS = [['cpi']]         # just need cpi (for inflation adjustment)

ACTION_DIMENSION = len(ALLOWED_ASSETS)
NUM_TRAJECTORIES = 10_000

# list of constant weight strategies: [stock_wt, bond_wt]
NUM_STRATEGIES = 10     # make it even

main(num_strategies=NUM_STRATEGIES,
     avg_len_list=[5, 10],
     probabilities_dict=probabilities_dict,
     num_trajectories=NUM_TRAJECTORIES)
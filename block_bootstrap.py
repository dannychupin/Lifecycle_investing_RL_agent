import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import floor

"""Data source: https://www.macrohistory.net/database/
    
    NOTES:
    -all returns NOMINAL (we will track cpi to turn these into REAL returns)
    -all returns in LOCAL CURRENCY
    -baseline for cpi is 1990
    
    WARNING: not all fields have data!
    -lots of `NaN` values, especially in times of war/economic crisis.
    ==> need to check for missing data in the block bootstrap simulation
"""

################
# I. LOAD DATA #
################

# set file path and load data
file_path = '/Users/danielchupin/Desktop/Rate_of_return_project/JSTdatasetR6.xlsx'
df = pd.read_excel(file_path)  # (2718, 59)

# collect relevant columns into three groups:
# (1) asset total returns: this is what we would like to use
# (2) asset total return proxies: in case total returns are not available
# (4) ancillary returns: change this to allow model to look at other macroeconomic variables

ASSET_RETURN_NAMES = ['eq_tr', 'housing_tr', 'bond_tr', 'bill_rate']
"""
    'eq_tr'         = equity total returns. Returns on a theoretical market cap-weighted stock index fund.
                    = 'eq_capgain' (price change) + 'eq_div_rtn' (dividend payout)
                    
    'housing_tr'    = housing total returns.  (assumes you own and rent out this real estate)
                    = 'housing_capgain' (price change) + 'housing_rent_rtn' (rent payout)
                    
    'bond_tr'       = bond total returns: returns on long-term (10+ years) government bonds
    
    'bill_rate'     = bill total returns: returns on short-term (2-6 months) government bonds
"""

# combine assets and their proxies
# the simulation will take the first value in each list that is available
# KEY: if at least one asset has all proxies 'NaN', SKIP THAT YEAR, AND SKIP TO NEXT COUNTRY

ASSET_PROXY_LIST = [['eq_tr', 'eq_capgain', 'eq_div_rtn'],
                    ['housing_tr', 'housing_capgain', 'housing_rent_rtn'],
                    ['bond_tr', 'bond_rate'],
                    ['bill_rate']]

# track macro variables, in particular cpi to calculate inflation rate, to turn NOMINAL into REAL returns
MACRO_LIST = [['gdp'], ['cpi']]     # CONVENTION: INFLATION IS ALWAYS LAST (it is most important)

############################
# II. CONSTRUCT SIMULATION #
############################

# hyperparameters for getting trajectories
AGE_AT_START = 30
RETIREMENT_AGE = 65
MU = 79.3
SIGMA = 32.8
SHAPE = 132.2

# change these in ablations
MIN_YEAR = 1871     # set to 1870+1 to calculate %change from prev year of macro variables (cpi, gdp, wages)
MAX_YEAR = 2020
AVG_RESIDENCE_LENGTH = 10  # MOST IMPORTANT. This is the mean of the geometric distribution

# change these in ablations to shape what kind of 'average investor' we are modeling
COUNTRIES = ['Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
             'Denmark', 'Spain', 'Finland', 'France', 'UK',
             'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
             'Portugal', 'Sweden', 'USA']

# for now: pick countries with uniform probability
COUNTRY_PROBABILITIES_UNIFORM = [1/len(COUNTRIES) for _ in range(len(COUNTRIES))]

# set seed for debugging
SEED = 41
np.random.seed(SEED)


class SimulatedInvestor:
    retirement_age = 65

    def __init__(self, min_year=MIN_YEAR,
                 max_year=MAX_YEAR,
                 countries=COUNTRIES,
                 country_probabilities=COUNTRY_PROBABILITIES_UNIFORM,
                 asset_proxy_list=ASSET_PROXY_LIST,
                 macro_list=MACRO_LIST,
                 age_at_start=AGE_AT_START,
                 avg_residence_length=AVG_RESIDENCE_LENGTH,
                 sigma=SIGMA,
                 mu=MU,
                 shape=SHAPE,
                 is_entropy_maximizing=False):

        self.min_year = min_year
        self.max_year = max_year

        self.age_at_start = age_at_start
        self.avg_residence_length = avg_residence_length
        self.is_entropy_maximizing = is_entropy_maximizing

        self.countries = countries
        self.country_probabilities = country_probabilities
        self.asset_proxy_list = asset_proxy_list
        self.macro_list = macro_list

        # combine the two lists to create `observation_list`
        self.observation_list = self.asset_proxy_list + self.macro_list
        self.observation_dimension = len(self.observation_list)     # what used to be called `state dimension`

        self.sigma = sigma
        self.mu = mu
        self.shape = shape

    def time_to_retirement(self, age: int):
        return min(self.retirement_age, age) - self.age_at_start

    def time_after_retirement(self, age: int):
        return max(age - self.retirement_age, 0)

    def generate_life_time(self):
        """
            T = age at death (int)

            Draw T from f(t) = Phi(y(t)), where
                Phi is standard normal, and
                y(t) = (t - mu) / (sigma * (1 - t/shape))

                mu = 79.3 = average life expectancy
                sigma = 32.8
                shape = 132.2 = asymptotic max

            Thus, f is a `skewed normal` distribution.

            See https://pmc.ncbi.nlm.nih.gov/articles/PMC3356396/

            Do it this way:
            - Sample Z standard normal
            - T = lambda * (sigma * Z + mu)/(sigma * Z + shape)

            Round T down
        """

        # keep generating until 't' is non-negative
        numerator = 0

        while numerator <= 0:
            z = np.random.standard_normal()
            numerator = self.sigma * z + self.mu
            denominator = self.sigma * z + self.shape

            if numerator > 0:  # implies denominator > 0
                t = self.shape * numerator / denominator
                return floor(t)

    def generate_time_to_retirement(self):
        t = 0

        while self.time_to_retirement(t) <= 0:
            # generate a new lifetime
            t = self.generate_life_time()
            if self.time_to_retirement(t) > 0:
                return self.time_to_retirement(t)

    def generate_time_after_retirement(self):
        t = 0

        while self.time_after_retirement(t) == 0:
            # generate a new lifetime
            t = self.generate_life_time()
            if self.time_after_retirement(t) > 0:
                return self.time_after_retirement(t)

    ################################################
    # MAXIMUM ENTROPY MODIFICATION OF TRAJECTORIES #
    ################################################
    # returns perturbed version of the trajectory that has, for each dimension,
    # the same relative temporal ordering of values.

    @staticmethod
    def resample_with_max_entropy(trajectory):  # `trajectory' is an np array of shape (N, D)
        # WARNING: method assumes no duplicate values per dimension!
        n, d = trajectory.shape

        if n <= 1:
            return trajectory

        perturbed_trajectory = np.zeros(trajectory.shape)

        for i in range(d):
            x = trajectory[:, i]  # has the time ordering
            x_sorted = np.sort(x)  # has the value ordering

            for t in range(n):
                t_order = np.where(x_sorted == x[t])[0][0]  # x_sorted[t_order] = x[t]
                left = 0
                right = n - 1

                # 1) set up bounds of uniform distribution
                if t_order == 0:
                    left = x_sorted[0] - (1 / 2) * (x_sorted[1] - x_sorted[0])
                    right = (1 / 2) * (x_sorted[0] + x_sorted[1])

                elif t_order == n - 1:
                    left = (1 / 2) * (x_sorted[-2] + x_sorted[-1])
                    right = x_sorted[-1] + (1 / 2) * (x_sorted[-1] - x_sorted[-2])

                else:
                    left = (1 / 2) * (x_sorted[t_order - 1] + x_sorted[t_order])
                    right = (1 / 2) * (x_sorted[t_order] + x_sorted[t_order + 1])

                x_hat_t = np.random.uniform(low=left, high=right)
                perturbed_trajectory[t, i] = x_hat_t
                if x_hat_t is None:
                    print(f'x_hat_t error! {x_hat_t}')

        return perturbed_trajectory

    ##########################
    # BLOCK BOOTSTRAP METHOD #
    ##########################
    def get_trajectory(self, total_time_steps: int):
        """
            'observation' = asset returns AND macro variable returns (used to be called 'state')
            if should_maximize_entropy == True, apply the static method above before returning
        """

        # returns sequence of observations (about 20 in length) for one investor's life
        time_series_of_observations = np.zeros(shape=(total_time_steps, self.observation_dimension), dtype=np.float32)
        time_elapsed = 0

        # repeatedly draw countries, and unroll their contiguous time series (`blocks`)
        while time_elapsed < total_time_steps:
            # draw initial year, from uniform (`high` is exclusive)
            t1 = np.random.randint(low=self.min_year, high=self.max_year)

            # draw length of block, from geometric distribution (with average equal 'avg_residence_length')
            time_in_country = np.random.geometric(p=1 / self.avg_residence_length)

            # truncate to prevent index overflow
            time_in_country = min(time_in_country, self.max_year - t1)

            # draw country from specified distribution
            country_name = np.random.choice(self.countries, size=1, p=self.country_probabilities)[0]

            # get returns and macro variables for country. Index by year for convenience
            country_data = df.iloc[df['country'] == country_name]
            country_data.set_index('year', inplace=True)

            # for breaking out of loop in case of missing data
            broke_inner = False

            # extract observations
            for t in range(0, time_in_country):
                year = t + t1

                for i, return_w_proxies in enumerate(self.observation_list):
                    # extract the row for the year
                    year_data_for_proxies = country_data.loc[year, return_w_proxies]

                    # extract name of column that has the first non-NaN value
                    first_non_nan_col_index = year_data_for_proxies.first_valid_index()

                    # if no valid proxy -> break out of both `for` loops, and select a different country
                    if first_non_nan_col_index is None:
                        broke_inner = True
                        break

                    # otherwise, fill in with the first value
                    value = country_data.loc[year, first_non_nan_col_index]
                    time_series_of_observations[time_elapsed, i] = value

                    # SPECIAL: turn absolute ancillary variables like cpi, gdp, debtgdp into returns
                    if return_w_proxies in self.macro_list:
                        # get name (they are 1-element arrays), and extract current and present values
                        measure_name = return_w_proxies[0]
                        value = country_data.loc[year, measure_name]
                        value_prev = country_data.loc[year - 1, measure_name]

                        # if value_previous is NaN, just say rtn is 0
                        if pd.isna(value_prev):
                            time_series_of_observations[time_elapsed, i] = 0
                        else:
                            rtn = (value - value_prev) / value_prev
                            time_series_of_observations[time_elapsed, i] = rtn

                if broke_inner:
                    break

                # if all data (up to proxy) for the year exists, investor has lived in this country for one year
                time_elapsed += 1
                if time_elapsed == total_time_steps:
                    break

        if self.is_entropy_maximizing:
            time_series_of_observations = self.resample_with_max_entropy(time_series_of_observations)

        return time_series_of_observations

    # synthetic validation regimes
    @staticmethod
    def get_synthetic_distribution(synthetic_mode):
        mu_1, mu_2, sigma_1, sigma_2, rho = 0, 0, 0, 0, 0

        if synthetic_mode == 'delta functions':
            # best strategy (simple rewards):....
            mu_1 = -0.4
            mu_2 = 0.1
            sigma_1 = 0
            sigma_2 = 0
            rho = 0

        elif synthetic_mode == 'correlated gaussians':
            # best constant strategy (simple rewards)
            mu_1 = 0.05
            mu_2 = 0.05
            sigma_1 = 0.2
            sigma_2 = 0
            rho = 1

        elif synthetic_mode == 'anticorrelated gaussians':
            # best constant strategy (simple rewards): 50% action_1
            mu_1 = 0.05
            mu_2 = 0.05
            sigma_1 = 0.2
            sigma_2 = 0.2
            rho = -1

        return mu_1, mu_2, sigma_1, sigma_2, rho

    def get_synthetic_trajectory(self, total_time_steps, synthetic_mode):
        mu_1, mu_2, sigma_1, sigma_2, rho = self.get_synthetic_distribution(synthetic_mode)

        mean = np.array([mu_1, mu_2])
        cov = np.array([[sigma_1 ** 2, rho * sigma_1 * sigma_2],
                        [rho * sigma_1 * sigma_2, sigma_2 ** 2]])

        time_series_of_observations = np.random.multivariate_normal(mean=mean,
                                                                    cov=cov,
                                                                    size=total_time_steps)  # (T, 2)

        return time_series_of_observations

###########################
# III. SOME ILLUSTRATIONS #
###########################


def sanity_check():
    # Sanity check: generate investors and lifetimes, and see if distributions look reasonable
    # NOTE: some artifacts with "spikes" in some bins. This is bin error/spillover, not simulation error!
    investor = SimulatedInvestor()
    num_investors = 10_000
    num_bins = 100
    investing_years = np.zeros(num_investors)
    retirement_years = np.zeros(num_investors)
    life_years = np.zeros(num_investors)

    for i in range(num_investors):
        life_years[i] = investor.generate_life_time()
        investing_years[i] = investor.generate_time_to_retirement()
        retirement_years[i] = investor.generate_time_after_retirement()

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
    ax[0].hist(life_years, bins=num_bins)
    ax[0].set_title('Lifespan')

    ax[1].hist(investing_years, bins=num_bins)
    ax[1].set_title('Number of investing years, from age 30 until retirement')

    ax[2].hist(retirement_years, bins=num_bins)
    ax[2].set_title('Number of retirement years (after 65)')
    fig.tight_layout()
    plt.show()


def generate_stories():
    investor = SimulatedInvestor()
    for _ in range(10):
        t = investor.generate_time_after_retirement()
        print('')
        print('===NEW TRAJECTORY===')
        print(f'Hi! I will live for {t} years after retirement.')
        print(investor.get_trajectory(t))

# sanity_check()
# generate_stories()

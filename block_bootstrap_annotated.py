import numpy as np
import pandas as pd

from math import floor

"""Data source: https://www.macrohistory.net/database/

    NOTES:
    -All returns NOMINAL (so, we would need to track inflation)
    -All returns in LOCAL CURRENCY
    -Baseline for CPI is 1990

    NOT ALL FIELDS HAVE DATA!
    -Lots of `NaN` values, especially in times of war/economic crisis.
    ==> Need to check for missing data in the block bootstrap
"""

### I. LOAD DATA ###

# Set file path and load data
file_path = '/Users/danielchupin/Desktop/Rate_of_return_project/JSTdatasetR6.xlsx'
df = pd.read_excel(file_path)  # (2718, 59)

# Collect relevant columns into three groups:
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

ASSET_PROXY_NAMES = ['eq_capgain', 'eq_div_rtn', 'housing_capgain', 'housing_rent_rtn', 'bond_rate']
ANCILLARY_VAR_NAMES = ['cpi', 'gdp', 'debtgdp', 'crisisJST']
STATE_NAMES = ASSET_RETURN_NAMES + ANCILLARY_VAR_NAMES

# Combine assets and their proxies. The simulation takes the first value available.
# E.g.: if eq_tr not available, see if eq_capgain is. If not, see if eq_div_rtn is.
# KEY: If there is one asset class for which no proxy is available, SKIP THAT YEAR, AND SKIP TO NEXT COUNTRY
PROXY_LIST = [['eq_tr', 'eq_capgain', 'eq_div_rtn'], ['housing_tr', 'housing_capgain', 'housing_rent_rtn'],
              ['bond_tr', 'bond_rate'], ['bill_rate'], ['cpi'], ['gdp'], ['debtgdp'], ['crisisJST']]

# PROXY_LIST[4] is ['cpi']. We will change this to inflation rate
# The only country with NaNs for CPI is Ireland 1870-1921

# RMK: can try to produce INTERNATIONAL stocks/bond/housing return data, by using the exchange rate to US 'xrusd'?
# Issue: would need to determine capitalization of each country's stock/bond market...
# ... or some other sensible weight for each country in an international index fund


### II. CONSTRUCT SIMULATION ###

# Hyperparameters for getting trajectories
AGE_AT_START = 30
RETIREMENT_AGE = 65
MU = 79.3
SIGMA = 32.8
SHAPE = 132.2

# Can change these in ablations (e.g. see if restriction to post-WWII would change things)
MIN_YEAR = 1871  # Set to 1871 in order to calculate inflation from CPI
MAX_YEAR = 2020
AVG_RESIDENCE_LENGTH = 10  # Avg contiguous length of time to spend in a country

# Can change these in ablations (e.g. to shape what kind of 'average investor' we are modeling)
COUNTRIES = ['Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
             'Denmark', 'Spain', 'Finland', 'France', 'UK',
             'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
             'Portugal', 'Sweden', 'USA']

ACTION_DIM = len(ASSET_RETURN_NAMES)  # 4 asset class weights
STATE_DIM = len(PROXY_LIST)  # 4 asset returns + 4 macro variables = 8

# Set seed for debugging
SEED = 42
np.random.seed(SEED)


class SimulatedInvestor:

    def __init__(self):
        self.name = 'Dubcek MacInnes'

    @staticmethod
    def time_to_retirement(age: int):
        return min(RETIREMENT_AGE, age) - AGE_AT_START

    @staticmethod
    def time_after_retirement(age: int):
        return max(age - RETIREMENT_AGE, 0)

    @staticmethod
    def generate_life_time():
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
            numerator = SIGMA * z + MU
            denominator = SIGMA * z + SHAPE

            if numerator > 0:  # implies denominator > 0
                t = SHAPE * numerator / denominator
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

    @staticmethod
    def get_trajectory(total_time_steps: int):
        ### BLOCK BOOTSTRAP METHOD ###

        # returns sequence of states (about 30 in length) for one investor's life

        time_series_of_states = np.zeros(shape=(total_time_steps, STATE_DIM))
        time_elapsed = 0

        # repeatedly draw countries, and unroll their contiguous time series (`blocks`)
        while time_elapsed < total_time_steps:
            # draw initial year, from uniform (`high` is exclusive)
            t1 = np.random.randint(low=MIN_YEAR, high=MAX_YEAR)

            # draw length of block, from geometric distribution (with average equal to AVG_SAMPLE_LENGTH)
            time_in_country = np.random.geometric(p=1 / AVG_RESIDENCE_LENGTH)

            # truncate to prevent index overflow
            time_in_country = min(time_in_country, MAX_YEAR - t1)

            # draw country, from uniform (?? or something else ??) on allowable countries
            country_idx = np.random.randint(low=0, high=len(COUNTRIES))
            country_name = COUNTRIES[country_idx]
            print(f'I am in {country_name}! For up to {time_in_country} years. The year is {t1}.')

            # get returns and macro variables for country. Index by year for convenience
            country_data = df.iloc[df['country'] == country_name]
            country_data.set_index('year', inplace=True)

            # for breaking out of loop in case of missing data
            broke_inner = False

            # extract states
            for t in range(0, time_in_country):
                # RMK: could just modify dataset to obviate checks for missing data...
                year = t + t1

                for i, return_w_proxies in enumerate(PROXY_LIST):
                    # extract the row for the year
                    year_data = country_data.loc[year, return_w_proxies]

                    # extract name of column that has the first non-NaN value
                    first_non_nan_col_index = year_data.first_valid_index()

                    # if no valid proxy -> break out of both `for` loops, and select a different country
                    if first_non_nan_col_index is None:
                        print('OOPS! Found a year with bad data. Moving to a new country...')
                        broke_inner = True
                        break

                    # otherwise, fill in with the first value
                    value = country_data.loc[year, first_non_nan_col_index]
                    time_series_of_states[time_elapsed, i] = value
                    if first_non_nan_col_index != return_w_proxies[0]:
                        print('*** Had to make a data substitution! ***')

                    # SPECIAL: update inflation info (i = 4)
                    if i == 4:
                        cpi = country_data.loc[year, 'cpi']
                        cpi_prev = country_data.loc[year - 1, 'cpi']
                        inflation_rate = 0

                        # if cpi_previous is NaN (only Ireland, with year-1 = 1921), just say inflation is 0
                        if pd.isna(cpi_prev):
                            time_series_of_states[time_elapsed, i] = inflation_rate

                        inflation_rate = (cpi - cpi_prev) / cpi_prev
                        time_series_of_states[time_elapsed, i] = inflation_rate

                if broke_inner:
                    break

                print(f'The time that has elapsed is {time_elapsed + 1}, still in {country_name}')

                # if all data (up to proxy) for the year exists, investor has lived in this country for one year
                time_elapsed += 1
                if time_elapsed == total_time_steps:
                    break

        return time_series_of_states


"""
investor = SimulatedInvestor()

# Sanity check: generate investors and lifetimes, and see if distributions look reasonable
num_investors = 10_000
investing_years = np.zeros(num_investors)
retirement_years = np.zeros(num_investors)

for i in range(num_investors):
    investing_years[i] = investor.generate_time_to_retirement()
    retirement_years[i] = investor.generate_time_after_retirement()

plt.hist(investing_years, bins=30)
plt.title('Number of investing years, from age 30 until retirement')
plt.show()

plt.hist(retirement_years, bins=30)     # This one has weird spikes at intervals of 5
plt.title('Number of retirement years (after 65)')
plt.show()


for _ in range(10):
    t = investor.generate_time_after_retirement()
    print('')
    print('===NEW TRAJECTORY===')
    print(f'Hi! I am {investor.name}. I will live for {t} years after retirement.')
    investor.get_trajectory(t)
"""
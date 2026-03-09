import pandas as pd
import matplotlib.pyplot as plt

"""Data source: https://www.macrohistory.net/database/

    NOTES:
    -All returns NOMINAL (so, we would need to track inflation)
    -All returns in LOCAL CURRENCY
    -Baseline for CPI is 1990

    NOT ALL FIELDS HAVE DATA!
    -Lots of `NaN` values, especially in times of war/economic crisis.
    ==> Need to check for missing data in the block bootstrap
"""

# Set file path and load data
file_path = '/Users/danielchupin/Desktop/Rate_of_return_project/JSTdatasetR6.xlsx'
df = pd.read_excel(file_path)  # (2718, 59)

# Display some basic info
with pd.option_context(
        "display.max_columns", None,  # show all columns
        "display.width", None,  # don't wrap based on terminal width
        "display.max_colwidth", None):  # don't truncate long cell contents
    print(df.head(10).to_string(index=True))

column_names = df.columns.tolist()
print(f'The columns are {column_names}')

countries = df['country'].unique()  # 18 countries
number_of_years = df['country'].value_counts()  # 151 each
print(countries)

# Graph a country's population and gdp
country = 'Sweden'
country_rows = df.iloc[df['country'] == country]
country_data = country_rows.loc[:, ['year', 'pop', 'gdp']]
country_data.set_index('year', inplace=True)

ax = country_data.plot(kind='line', y='pop', color='blue', legend=True)
country_data.plot(kind='line', y='gdp', color='red', secondary_y=True, ax=ax, legend=True)
ax.set_ylim(bottom=0)
ax.right_ax.set_ylim(bottom=0)

ax.set_ylabel('Population')
ax.right_ax.set_ylabel('GDP')
plt.title(f'GDP and Population for {country}')
plt.show()

# show CPI
no_cpi_rows = df.iloc[df['cpi'].isna()]
print(no_cpi_rows.loc[:, ['year', 'country', 'cpi']])   # it's Ireland, 1870-1921
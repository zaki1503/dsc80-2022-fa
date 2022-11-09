# lab.py


import pandas as pd
import numpy as np
import io
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def latest_login(login):
    """Calculates the latest login time for each user
    :param login: a DataFrame with login information
    :return: a DataFrame with latest login time for
    each user indexed by "Login Id"
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> result = latest_login(login)
    >>> len(result)
    433
    >>> result.loc[393, "Time"] > 9
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    """
    Calculates the the login frequency for each user.
    :param login: a DataFrame with login information but without unique IDs
    :return: a Series, indexed by Login ID, containing 
    the login frequency for each user.
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> freq = count_frequency(login)
    >>> len(freq)
    433
    >>> np.isclose(freq.loc[466], 0.24517906336088155)
    True
    """
    login = login
    login['Time'] = login['Time'].apply(pd.to_datetime)
    login

    time_mask = (login['Time'].dt.hour >= 16) & \
                (login['Time'].dt.hour <= 20)

    df1 = login[time_mask].groupby('Login Id').count()
    df2 = login.groupby('Login Id').count()
    df2['Time'] = 0
    df2.rename(columns={'Time':'nottime'}, inplace=True)
    df2

    out = pd.concat([df1, df2], axis = 1).drop(columns= ['nottime']).fillna(0)
    return out

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def count_frequency(login):
    """
    Calculates the the login frequency for each user.
    :param login: a DataFrame with login information but without unique IDs
    :return: a Series, indexed by Login ID, containing 
    the login frequency for each user.
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> freq = count_frequency(login)
    >>> len(freq)
    433
    >>> np.isclose(freq.loc[466], 0.24517906336088155)
    True
    """
    login = login
    login['Time'] = login['Time'].apply(pd.to_datetime)

    users = login.groupby('Login Id').min()
    users['today'] = '2018-01-05 00:00:00'
    users = users.apply(pd.to_datetime)

    users['freq'] = (users['today'] - users['Time'])
    users['freq'] = np.int64(users['freq']/np.timedelta64(1, 'D'))
    users['freq'] = login.groupby('Login Id').count()['Time']/users['freq']
    return users['freq']

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def total_seller(sales):
    """
    total_seller should take in the sales DataFrame and 
    return a DataFrame that contains the total sales 
    for each seller, indexed by 'Name'. There should not be any NaNs.

    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = total_seller(sales)
    >>> out.shape[0]
    3
    >>> out["Total"].sum() < 15000
    True
    """
    return sales.groupby('Name').sum()


def product_name(sales):
    """
    product_name should take in the sales DataFrame and 
    return a DataFrame that contains the total sales 
    for each seller, indexed by 'Product'. 
    Do not fill in NaNs.
    
    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = product_name(sales)
    >>> out.size
    15
    >>> out.loc["pen"].isnull().sum()
    0
    """
    out = pd.pivot_table(
                        data=sales, 
                        columns=['Name'], 
                        index='Product', 
                        aggfunc=np.sum
                        )
    out.columns= out.columns.droplevel(0)
    return out


def count_product(sales):
    """
    count_product should take in the sales DataFrame and 
    return a DataFrame that contains the total number of 
    items sold product-wise and name-wise per date. 
    Replace NaNs with 0s.

    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = count_product(sales)
    >>> out.loc["boat"].loc["Trump"].value_counts()[0]
    6
    >>> out.size
    70
    """
    out = pd.pivot_table(
                data=sales, 
                columns=['Date'], 
                index=['Product', 'Name'], 
                aggfunc=np.sum, 
                fill_value=0
                )
    out.columns= out.columns.droplevel(0)
    return out


def total_by_month(sales):
    """
    total_by_month should take in the sales DataFrame 
    and return a pivot table that contains the total 
    sales name-wise, product-wise per month. 
    Replace NaNs with 0s.
    
    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = total_by_month(sales)
    >>> out["May"].idxmax()
    ('Smith', 'book')
    >>> out.shape[1]
    5
    """
    sales['Date'] = sales['Date'].apply(pd.to_datetime)
    sales

    sales['Date'] = sales['Date'].dt.month_name().astype(str)
    sales

    out = pd.pivot_table(
                    data=sales, 
                    columns = ['Date'], 
                    index=['Name', 'Product'], 
                    aggfunc=np.sum, 
                    fill_value=0
                    )

    out.columns= out.columns.droplevel(0)
    return out


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def diff_of_median_proportions(data, col='orange'):
    """
    diff_of_median_proportions takes in a DataFrame like skittles 
    and returns the absolute difference of median proportions 
    between the number of oranges per bag from Yorkville and Waco.
    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = diff_of_median_proportions(skittles)
    >>> 0 <= out
    True
    """
    df = data
    col_lst = list(data.columns)
    col_lst.remove('Factory')
    col_lst

    if col+'_prop' not in df.columns:
        df['total'] = df[col_lst].sum(axis=1)
        df[col+'_prop'] = df[col]/df['total']
        df.groupby('Factory').median()[col+'_prop']

    wacoprop = df.groupby('Factory').median()[col+'_prop'].loc['Waco']
    yvprop = df.groupby('Factory').median()[col+'_prop'].loc['Yorkville']

    return np.abs(wacoprop - yvprop)


def simulate_null(data, col='orange'):
    """
    simulate_null takes in a DataFrame like skittles and 
    returns one simulated instance of the test statistic 
    under the null hypothesis.
    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = simulate_null(skittles)
    >>> isinstance(out, float)
    True
    >>> 0 <= out <= 1.0
    True
    """
    df = data
    df['Factory'] = np.random.permutation(df['Factory'])
    
    col_lst = list(data.columns)
    col_lst.remove('Factory')

    df['prop'] = df[col]/df[col_lst].sum(axis=1)
    
    waco = df.groupby('Factory').median()['prop'].loc['Waco']
    yv = df.groupby('Factory').median()['prop'].loc['Yorkville']

    return np.abs(waco-yv)


def pval_color(data, col='orange'):
    """
    pval_color takes in a DataFrame like skittles and 
    calculates the p-value for the permutation test 
    using 1000 trials.
    
    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> pval = pval_color(skittles)
    >>> isinstance(pval, float)
    True
    >>> 0 <= pval <= 0.2
    True
    """
    obs = diff_of_median_proportions(data,col)
    trials = []
    for i in range(1000):
        trial = simulate_null(data, col)
        trials.append(trial)

    trials = np.array(trials)

    p = np.count_nonzero(trials >= obs)/1000

    return p


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def ordered_colors():
    """
    ordered_colors returns your answer as an ordered
    list from "most different" to "least different" 
    between the two locations. You list should be a 
    hard-coded list, where each element has the 
    form (color, p-value).

    :Example:
    >>> out = ordered_colors()
    >>> len(out) == 5
    True
    >>> colors = {'green', 'orange', 'purple', 'red', 'yellow'}
    >>> set([x[0] for x in out]) == colors
    True
    >>> all([isinstance(x[1], float) for x in out])
    True
    """
    out = [
        ('red', 0.001),
        ('purple', 0.054),
        ('yellow', 0.128),
        ('orange', 0.327),
        ('green', 0.577)
        ]
    return out


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------



def same_color_distribution():
    """
    same_color_distribution outputs a hard-coded tuple 
    with the p-value and whether you 'Fail to Reject' or 'Reject' 
    the null hypothesis.

    >>> out = same_color_distribution()
    >>> isinstance(out, tuple)
    True
    >>> isinstance(out[0], float)
    True
    >>> out[1] in ['Fail to Reject', 'Reject']
    True
    """
    return (0.013, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def perm_vs_hyp():
    """
    Multiple choice response for Question 8.

    >>> out = perm_vs_hyp()
    >>> ans = ['P', 'H']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """
    out = [
        'H',
        'H',
        'P',
        'P',
        'H'
    ]
    return out


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def after_purchase():
    """
    Multiple choice response for question 8

    >>> out = after_purchase()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """
    return ['MCAR', 'MAR', 'MD', 'MAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def multiple_choice():
    """
    Multiple choice response for question 9

    >>> out = multiple_choice()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    >>> out[1] in ans
    True
    """
    return ['NI', 'MAR', 'MD', 'MCAR', 'MCAR']

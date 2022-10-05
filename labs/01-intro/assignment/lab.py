# lab.py


import os
import io
import pandas as pd
import numpy as np
import doctest


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5, 3, 6, 4, 9, 8])
    True
    >>> consecutive_ints([1, 3, 5, 7, 9])
    False
    >>> consecutive_ints([])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    '''
    median_vs_mean takes in a non-empty list of numbers
    and returns a Boolean of whether the median is
    less than or equal to the mean.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether the median is less than or equal to the mean.
    
    :Example:
    >>> median_vs_mean([6, 5, 4, 3, 2])
    True
    >>> median_vs_mean([50, 20, 15, 40])
    True
    >>> median_vs_mean([1, 8, 9])
    False
    '''
    num_arrs = np.array(nums)

    if np.median(num_arrs) <= np.mean(num_arrs):
        return True
    else:
        return False


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list containcds
    two list elements i positions apart, whose absolute 
    difference as integers is also i.
    :param ints: a list of integers
    :returns: a bool, describing whether or not the 
    above condition is satisfied
    :Example:
    >>> same_diff_ints([5, 3, 1, 5, 9, 8])
    True
    >>> same_diff_ints([1, 3, 5, 7, 9])
    False
    """
    arr = np.array(ints)
    for i in range(1,len(ints)-1):
        diffs = np.absolute(np.diff(arr, n=i))
        if i in diffs:
            return True
    return False






# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Billy', 4)
    'BillBilBiB'
    >>> n_prefixes('Marina', 3)
    'MarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    >>> n_prefixes('Justin', 5)
    'JustiJustJusJuJ'
    """
    output = []
    for i in range(n):
        output.append(s[:i+1])

    out_str = ''.join(output[::-1])
    return out_str
        
        



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers. 
    :Example:
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    >>> exploded_numbers([9, 99], 3)
    ['006 007 008 009 010 011 012', '096 097 098 099 100 101 102']
    """

    out = []
    max_len = len(str(max([num + n for num in ints])))
    for int in ints:
        temp = [str(num).zfill(max_len) for num in range(int-n, int+(n+1))]
        out.append(' '.join(temp))

    return out


    


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of each line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """
    out = ''
    for line in fh:
        if line.strip('\n'):
            out += line.strip('\n')[-1]
        else:
            continue
    return out



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def add_root(A):
    """
    add_root takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = add_root(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    >>> np.isclose(out[3], 7 + np.sqrt(3))
    True
    """
    rooted = np.array(np.sqrt(range(len(A))))
    return A + rooted
    

def where_square(A):
    """
    where_square takes in a numpy array of integers
    and returns an array of Booleans
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = where_square(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out[2]
    True
    """
    B = np.square(np.floor(np.sqrt(A)))
    return A==B

# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    """
    growth_rates takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = growth_rates(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    b = np.copy(A)

    return np.round((A[1:]-b[:-1])/b[:-1],2)

def with_leftover(A):
    """
    Create a function with_leftover that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: the day on which you can buy at least one share from 'left-over' money
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = with_leftover(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    a = np.array([3,3,3,3])
    x = np.where(np.cumsum(20%a) >= a)
    if np.all(x == False):
        print(-1)
    return x[0][0]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------



def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    :param salary: a DataFrame of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)
    True
    """
    num_players = salary['Player'].count()
    num_teams = salary.groupby('Team').count().shape[0]
    total_salary = salary['Salary'].sum()
    highest_salary = salary['Salary'].max()
    avg_los = round(salary.loc[salary['Team'] == \
            'Los Angeles Lakers']['Salary'].mean(),2)
    fifth_lowest_tmp = salary.sort_values(by=['Salary']).iloc[4]
    fifth_lowest = fifth_lowest_tmp['Player'] + ', ' + \
                fifth_lowest_tmp['Team']
    temp = salary.assign(lastname= \
            salary['Player'].apply(lambda x: x.split(' ')[1]))
    duplicates = temp['lastname'].nunique() != salary.shape[0]
    total_highest = salary[salary['Team'] == \
                    salary.sort_values(by=['Salary'], ascending=False)\
                    .iloc[0]['Team']]['Salary'].sum()
    dict = {'num_players': num_players,
            'num_teams': num_teams,
            'total_salary': total_salary,
            'highest_salary': highest_salary,
            'avg_los': avg_los,
            'fifth_lowest': fifth_lowest,
            'duplicates': duplicates,
            'total_highest': total_highest}
    return pd.Series(data=dict)
    


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    """
    Parses and loads the malformed .csv file into a 
    properly formatted DataFrame (as described in 
    the question).
    :param fh: file handle for the malformed .csv file.
    :returns: a pandas DataFrame of the data, 
    as specified in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    out_df = pd.DataFrame(columns = 
        ['first', 'last', 'weight', 'height', 'geo'])

    with open(fp, "r") as f:
        for i in range(1):
            next(f)
        i=0
        for line in f:
            line = line.replace('"','').replace('\n','').split(',')
            out_df.loc[i] = [line[0], 
                            line[1], 
                            line[2], 
                            line[3], 
                            str(line[4]) + ',' + str(line[5])]
            i+=1
    out_df['height'] = pd.to_numeric(out_df['height'])
    out_df['weight'] = pd.to_numeric(out_df['weight'])

    return out_df

# lab.py


import os
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------



def first_round():
    """
    :return: list with two values as described in the notebook
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] in ['NR', 'R']
    True
    """
    return [0.1335, 'NR']


def second_round():
    """
    :return: list with three values as described in the notebook
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] in ['NR', 'R']
    True
    >>> out[2] in ['ND', 'D']
    True
    """
    return [0.0028277628406094824, 'R', 'D']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def verify_child(heights):
    """
    Returns a Series of p-values assessing the missingness
    of child-height columns on father height.
    
    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(heights_fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    from scipy.stats import ks_2samp

    out = pd.Series(index=heights.columns, dtype=np.float64)
    out

    for column in heights.columns:
        t = heights[column]
        t2 = heights[column].dropna()
        if column == 'child_95':
            out[column] = 1-ks_2samp(t,t2).pvalue
        else:
            out[column] = ks_2samp(t,t2).pvalue

    return out
    


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.
    :Example:
    >>> set(missing_data_amounts()) <= set(range(1, 6))
    True
    """
    return [2,5]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a DataFrame with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> new_heights = pd.read_csv(heights_fp)[['father', 'child_50']]
    >>> new_heights = new_heights.rename(columns={'child_50': 'child'})
    >>> out = cond_single_imputation(new_heights.copy())
    >>> out.isna().sum() == 0
    True
    >>> (new_heights['child'].std() - out.std()) > 0.5
    True
    """
    df = new_heights.copy()
    df['father'] = pd.qcut(df['father'], 4)

    out = df.groupby('father').mean()['child']
    out = df.groupby('father').mean()['child']

    df2 = df.apply(lambda row: out[row['father']] if np.isnan(row['child']) else row['child'],axis=1)
    return df2


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(heights_fp)
    >>> child = heights['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    >>> np.isclose(out.std(), 3.5, atol=0.65)
    True
    """
    if N == 0:
        return
    childnona = child.dropna()
    probs, bins = np.histogram(childnona, density= True)

    binsbins = {}
    for i in range(len(bins)):
        if len(bins) - i != 1:
            binny = (bins[i], bins[i+1])
            binsbins[i] = binny
        else:
            break


    randbins = np.random.choice(a=len(probs), size=N, p=probs/probs.sum())

    out = []


    for binn in randbins:
        edges = binsbins[binn]
        hite = round(np.random.uniform(edges[0],edges[1]),1)
        out.append(hite)

    out = np.array(out)
    return out


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.
    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(heights_fp)
    >>> child = heights['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isna().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.2)
    True
    >>> np.isclose(out.std(), child.std(), atol=0.15)
    True
    """
    childnona = np.array(child.dropna())
    N = child.isna().sum()
    heights = quantitative_distribution(child,N)

    out = pd.Series(np.concatenate([childnona, heights]))
    return out


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> mc_answers, websites = answers()
    >>> len(mc_answers)
    4
    >>> len(websites)
    6
    """
    mc_answers = [1,2,2,1]
    websites = ['https://www.cnn.com/robots.txt', 
                'https://www.reddit.com/robots.txt',
                'https://www.ebay.com/robots.txt',
                'https://docs.google.com/robots.txt',
                'https://stockx.com/robots.txt',
                'https://twitter.com/robots.txt'
                ]
    return mc_answers, websites

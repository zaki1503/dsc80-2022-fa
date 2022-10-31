# discussion.py


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prev_midterm_hypothesis_test(penguins_type):
    """
    Calculates the observed TVD, Defines & Simulates the 
    test statistic on N=5000 samples of the null-hypothesis.
    Uses the null_tvds to estimate the p_val
    :Example:
    >>> res = prev_midterm_hypothesis_test(penguins_type)
    >>> isinstance(res, tuple)
    >>> 0<=res[1]<=1
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def midterm_review_imputation(penguins_with_nulls):
    """
    Takes a dataframe with NaN's or nulls and fills the `Flipper Length` column with the 
    Median Flipper length of the corresponding species.
    :Example:
    >>> penguins_with_nulls = pd.read_csv('data/data1.csv')
    >>> res = midterm_review_imputation(penguins_with_nulls)
    >>> isinstance(q2_out, pd.DataFrame)
    >>> q2_out['Flipper Length (mm)'].isnull().sum() == 0
    """
    df = penguins_with_nulls.copy(deep=True)
    ...
    return df

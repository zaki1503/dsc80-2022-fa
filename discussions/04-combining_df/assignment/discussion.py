# discussion.py


import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def specific_combined_seasons(df1, df2):
    """
    Create a function that return, as a tuple, a dataframe combining
    the 2017 and 2018 MLB seasons as per the consitions specified as 
    well as the highest average R/G combining both the seasons.

    :Example:
    >>> mlb_2017 = pd.read_csv(os.path.join('data','mlb_2017.txt'))
    >>> mlb_2018 = pd.read_csv(os.path.join('data','mlb_2018.txt'))
    >>> result = combined_seasons(mlb_2017, mlb_2018)
    >>> result[0].shape
    (30, 8)
    >>> result[1] in result[0].index
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def plot_meal_by_day(tips):
    """
    Plots the counts of meals in tips by day.
    plot_meal_by_day returns a Figure
    object; your plot should look like the plot in the notebook.

    :Example:
    >>> tips = sns.load_dataset('tips')
    >>> fig = plot_meal_by_day(tips)
    >>> type(fig)
    <class 'matplotlib.figure.Figure'>
    """
    fig = plt.figure()
    ...
    return fig


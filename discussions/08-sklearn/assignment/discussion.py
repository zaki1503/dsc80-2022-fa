# discussion.py


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class LowStdColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=0):
        '''
        Drops columns whose standard deviation is less than thresh.
        '''
        self.thresh = thresh

    def fit(self, X, y=None):
        """
        ...
        """
        # self.columns_ = ...
        # return self
        
        ...

    def transform(self, X, y=None):
        """
        
        ...
        """
        ...

# lab.py


import pandas as pd
import numpy as np
import os
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def simple_pipeline(data):
    '''
    simple_pipeline takes in a dataframe like data and returns a tuple
    consisting of the pipeline and the predictions your model makes
    on data (as trained on data).
    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = simple_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], FunctionTransformer)
    True
    >>> preds.shape[0] == data.shape[0]
    True
    '''
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    log = FunctionTransformer(np.log)

    train = np.array(data['c2']).reshape(-1, 1)
    y = np.array(data['y']).reshape(-1, 1)
    spl = Pipeline([('log', log),('lin-reg', lr)])
    fitspl = spl.fit(train, y)
    pred = spl.predict(np.array(data[['c2']]).reshape(-1,1))

    return (fitspl, pred.flatten())

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multi_type_pipeline(data):
    '''
    multi_type_pipeline that takes in a dataframe like data and
    returns a tuple consisting of the pipeline and the predictions
    your model makes on data (as trained on data).
    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = multi_type_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], ColumnTransformer)
    True
    >>> data.shape[0] == preds.shape[0]
    True
    '''
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    oneh = OneHotEncoder()
    log = FunctionTransformer(np.log)

    data_f = data.drop('y', axis=1)

    ct = ColumnTransformer(
        transformers = [('log', log, ['c2']),('cat', oneh, ['group'])],
        remainder="passthrough"
    )


    mpl = Pipeline([
        ("columntransform", ct),
        ('lin-reg', lr)
                    ])

    outfit = mpl.fit(data_f, data['y'])
    outpred = mpl.predict(data_f)

    return (outfit, outpred)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


# Imports
from sklearn.base import BaseEstimator, TransformerMixin

class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X might not be a pandas DataFrame (e.g. a np.array)
        df = pd.DataFrame(X)

        # Compute and store the means/standard-deviations for each column (e.g. 'c1' and 'c2'), 
        # for each group (e.g. 'A', 'B', 'C').  
        # (Our solution uses a dictionary)
        grplist = list(df.iloc[:,0].value_counts().index)

        dct = {}
        for grp in grplist:
            tdf = df[df.iloc[:,0]==grp]
            dct[grp] = (tdf.mean(numeric_only=True),tdf.std(numeric_only=True))

        self.grps_ = dct
        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """
        import pandas as pd

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        
        # Hint: Define a helper function here!

        out = pd.DataFrame(X)
        
        cols = list(self.grps_.values())[0][0].index.to_list()

        def helper(row):
            val = row[col]
            mu = self.grps_[row.iloc[0]][0][col]
            std = self.grps_[row.iloc[0]][1][col]
            return (val - mu)/ std

        for col in cols:
            out[col] = out.apply(helper, axis=1)

        return out.set_index(out.columns[0])


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def eval_toy_model():
    """
    Hard-coded answers to Question 4.
    :Example:
    >>> out = eval_toy_model()
    >>> len(out) == 3
    True
    >>> np.all([len(t) == 2 for t in out])
    True
    """
    out = [
        (2.7551086974518104, 0.39558507345910765),
        (2.3148336164355268, 0.573324931567333),
        (2.3524664344350126, 0.5729929650348398)
    ]

    return out


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def rmse(true, pred):
    from sklearn.metrics import mean_squared_error
    import math
    return math.sqrt(mean_squared_error(true,pred))

def tree_reg_perf(galton):
    """

    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = tree_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    >>> out['train_err'].iloc[-1] < out['test_err'].iloc[-1]
    True
    """
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    df = galton

    # features
    X = df.drop('childHeight', axis=1)
    # outcome
    y = df.childHeight

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 
    
    out=pd.DataFrame(columns=['train_err', 'test_err'],index=list(range(1,21)))

    for i in range(1,21):
        clf = DecisionTreeRegressor(max_depth=i)
        clf.fit(X_train, y_train)
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        j= i-1
        out.iloc[j] = (rmse(y_train, train_preds), rmse(y_test, test_preds))

    out = out.astype(np.float64)

    return out

def knn_reg_perf(galton):
    """
    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = knn_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    """
    # Add your imports here
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    df= galton

    # features
    X = df.drop('childHeight', axis=1)
    # outcome
    y = df.childHeight

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 
    
    out=pd.DataFrame(columns=['train_err', 'test_err'],index=list(range(1,21)))

    for i in range(1,21):
        clf = KNeighborsRegressor(n_neighbors=i)
        clf.fit(X_train, y_train)
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        j= i-1
        out.iloc[j] = (rmse(y_train, train_preds), rmse(y_test, test_preds))

    out = out.astype(np.float64)

    return out


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def titanic_model(titanic):
    """
    :Example:
    >>> fp = os.path.join('data', 'titanic.csv')
    >>> data = pd.read_csv(fp)
    >>> pl = titanic_model(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> from sklearn.base import BaseEstimator
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> preds = pl.predict(data.drop('Survived', axis=1))
    >>> ((preds == 0)|(preds == 1)).all()
    True
    """
    # Add your import(s) here
    from sklearn.preprocessing import StandardScaler, Binarizer
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    df = titanic.copy()

    def fare_cats(fares):
        def inner(fare):
            if fare <= 7.8:
                return 0
            elif fare <= 14.4:
                return 1
            elif fare <= 31:
                return 2
            else:
                return 3
        return pd.Series(fares).apply(inner).values

    def title(names):
        def inner(name):
            marital = ['Mr. ', 'Mrs. ']
            vip = ["Sir. ","Lady ","Major ","Don ","Dr. ","Col. "]
            if any(st in name for st in marital):
                return 1
            elif any(st in name for st in vip):
                return 2
            else:
                return 0
        return pd.Series(names).apply(inner).values

    df = df.replace({"male":0, "female":1})


    df['Name']  = title(df.Name)
    df['Fare'] = fare_cats(df.Fare)


    titler = Pipeline([('tit',FunctionTransformer(title, validate=False)),
                        ('1h',OneHotEncoder(handle_unknown='ignore'))
                    ])
    onehotdirect = Pipeline([('1h',OneHotEncoder(handle_unknown='ignore'))])
    age_st = Pipeline([('ssbg', StdScalerByGroup())])
    fr = Pipeline([('fare', FunctionTransformer(fare_cats))])

    middleman = ColumnTransformer(
        transformers=[
                    ('1hd', onehotdirect, ['Pclass']),
                    ('z', age_st, ['Pclass', 'Age'])
        ],
        remainder='passthrough'
    )

    pre_out = Pipeline([
                ('mm', middleman),
                ('classifier', DecisionTreeClassifier(max_depth=8))])

    out = pre_out.fit(df.drop('Survived', axis=1), df.Survived)
    return out


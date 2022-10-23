# lab.py


import os
import io
from typing_extensions import reveal_type
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1, 7))
    True
    """
    return (3,5)


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1, 7))
    True
    """
    return(2,6)

def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1, 5))
    True
    """
    return (1,4)

def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in set(range(1, 6))
    True
    """
    return 3


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def clean_universities(df):
    """ 
    clean_universities takes in the raw rankings DataFrame
    and returns a cleaned DataFrame according to the instructions
    in the lab notebook.
    >>> fp = os.path.join('data', 'universities_unified.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_universities(df)
    >>> cleaned.shape[0] == df.shape[0]
    True
    >>> cleaned['nation'].nunique() == 59
    True
    """
    datatypedict = {
    'broad_impact': np.int64,
    'national_rank_cleaned': np.int64
                }

    dfc = df
    dfc['institution'] = dfc['institution'].replace(r'\n',', ', regex=True)
    dfc['institution'] = dfc['institution'].replace(r'\r,',',', regex=True)
    dfc['national_rank_cleaned'] = dfc['national_rank'].apply(lambda x: x.split(', ')[1])

    dfc['nation'] = dfc['national_rank'].apply(lambda x: x.split(', ')[0])

    dfc= dfc.replace('Czechia', 'Czech Republic')
    dfc= dfc.replace('USA', 'United States')
    dfc= dfc.replace('UK', 'United Kingdom')
    dfc = dfc.astype(datatypedict)

    def r_1_pub(row):
        if pd.isnull(row['control']) \
            or pd.isnull(row['city']) \
            or pd.isnull(row['state']):
            return False
        elif row['control'] == 'Public':
            return True
        else: 
            return False

    dfc['is_r1_public'] = dfc.apply(r_1_pub, axis=1)
    dfc = dfc.drop(columns=['national_rank'])
    return dfc

def university_info(cleaned):
    """
    university_info takes in a cleaned rankings DataFrame
    and returns a list containing the four values described
    in the lab notebook.
    >>> fp = os.path.join('data', 'universities_unified.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_universities(df)
    >>> info = university_info(cleaned)
    >>> len(info) == 4
    True
    >>> all([isinstance(x, y) for x, y in zip(info, [str, float, int, str])])
    True
    >>> (info[1] >= 0) & (info[1] <= 1)
    True
    """
    dfc = cleaned
    boolstates = pd.Series(dfc[pd.notnull(dfc['state'])].groupby('state')\
                        .count()['institution'] >= 3)
    statelist = boolstates[boolstates== True].index

    L = dfc[dfc['state'].isin(statelist.to_list())].\
            groupby('state').mean()['score'].idxmin()

    propfac = dfc[(dfc['world_rank'] <= 100) & 
                (dfc['quality_of_faculty'] <= 100)].shape[0]/100

    L2 = dfc[dfc['national_rank_cleaned'] == 1].\
            set_index('institution')['world_rank'].idxmax()

    dfc[pd.notnull(dfc['state'])].groupby(['state', 'control']).count()['institution'].reset_index()

    dfc2 = dfc[pd.notnull(dfc['state'])]
    ser = dfc2['control'].apply(lambda x: 0 if x == 'Public' else 1)
    dfc2 = dfc2.assign(control= ser)

    priv50plus = pd.Series(dfc2.groupby('state').mean()['control'] >= 0.5)
    privstatenum = priv50plus[priv50plus].size

    return [L, propfac, privstatenum,L2]



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def std_scores_by_nation(cleaned):
    """
    std_scores_by_nation takes in a cleaned DataFrame of university rankings
    and returns a DataFrame containing standardized scores, according to
    the instructions in the lab notebook.
    >>> fp = os.path.join('data', 'universities_unified.csv')
    >>> play = pd.read_csv(fp)
    >>> cleaned = clean_universities(play)
    >>> out = std_scores_by_nation(cleaned)
    >>> out.shape[0] == cleaned.shape[0]
    True
    >>> all(out.columns == ['institution', 'nation', 'score'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    dfc3 = cleaned[['institution', 'nation', 'score']]

    natsdevs = dfc3.groupby('nation').std(ddof=0)
    natmeans = dfc3.groupby('nation').mean()

    def standardize_score(row):
        nation = row['nation']
        score = row['score']
        std = natsdevs.loc[nation]
        mean = natmeans.loc[nation]
        out = (score-mean)/std
        return out
    copy = dfc3.copy()
    dfc3 = dfc3.assign(score = copy.apply(standardize_score, axis=1))
    return dfc3

    

def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0] in np.arange(1, 4)
    True
    >>> isinstance(out[1], str)
    True
    """
    return [2, 'Malaysia']


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    """
    read_linkedin_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_linkedin_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_linkedin_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    csvdirlist = os.listdir(dirname)
    big = pd.DataFrame(pd.read_csv(os.path.join(dirname, csvdirlist[0])))
    big['fullname'] = big['first name'] + ' ' + big['last name']
    big
    for csv in csvdirlist[1:]:
        df = pd.read_csv(os.path.join(dirname, csv))
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace('_', ' ')
        df['fullname'] = df['first name']  + ' ' +  df['last name']
        big = pd.concat([big, df], axis=0)
    
    big = big.set_index('fullname')
    big = big.fillna('')
    out = big[[
            'first name', 
            'last name', 
            'current company', 
            'job title', 
            'email', 
            'university'
            ]]
    return out


def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a hardcoded list of answers to the problems in the notebook
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_linkedin_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> isinstance(out[0], float)
    True
    >>> isinstance(out[1], int)
    True
    >>> isinstance(out[2], str)
    True
    """
    ohionurses = df[(df['university'].str.contains('Ohio'))
        & (df['job title'].str.contains('Nurse'))
        ].shape[0]/df[df['university'].str.contains('Ohio')].shape[0]
    numeng = df[df['job title'].str.endswith('Engineer')].shape[0]
    idxlong = pd.DataFrame(
        columns = [''],
        data= df['job title'].apply(lambda x: len(x)).to_list()).idxmax()
    long = df.iloc[int(idxlong)]['job title']
    man = df['job title'].apply(str.lower)
    numman = [man[:].str.contains('manager')].shape[0]

    return [ohionurses, numeng, long, numman]



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    """
    read_student_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 1-1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = read_student_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> read_student_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    survdir=  dirname
    surveys = os.listdir(survdir)
    names = pd.DataFrame(pd.read_csv(os.path.join(survdir, surveys[0])))
    names
    cols = [x[:-4] for x in surveys][1:]
    df = pd.DataFrame(index= names['id'], columns=['name'] + cols)
    df['name'] = names['name']
    df

    for survey, col in zip(surveys[1:], cols):
        temp = pd.DataFrame(pd.read_csv(os.path.join(survdir, survey)))
        temp = temp.set_index('id')
        df[col] = temp[temp.columns[0]]

    return df


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 1-1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = read_student_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    >>> out['ec'].max()
    6
    """
    cols = df.columns.to_list().remove('name')
    outdf = df[cols].notnull().astype('int')
    outdf['ec'] = outdf.apply(sum, axis=1)

    i = 0
    for col in cols:
        while i < 2:
            if outdf[col].sum()/outdf[col].shape[0]:
                outdf['ec'] += 1
                i+=1

    outdf['name'] = df['name']
    return outdf[['name', 'ec']]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    """
    most popular 'ProcedureType'
    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = most_popular_procedure(pets, procedure_history)
    >>> isinstance(out, str)
    True
    """
    pets = pd.read_csv('data/pets/Pets.csv')
    procedure_history = pd.read_csv('data/pets/ProceduresHistory.csv')
    temp = pets.merge(procedure_history, right_on= 'PetID', 
                left_on='PetID')[['PetID', 'ProcedureType']].set_index('PetID')

    procedure_history = procedure_history.set_index('PetID')

    out = pd.concat([procedure_history, temp])['ProcedureType']\
        .value_counts(0).idxmax()
    return out

def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    multipletowners = set(pets[pets['OwnerID'].duplicated(keep=False)]['OwnerID'].to_list())
    owners = owners.set_index('OwnerID')

    out = pd.Series(dtype='object')
    out

    pets[pets['OwnerID']==5508]['Name'].to_list()

    def helper(key):
        if key in multipletowners:
            return (owners.loc[key]['Name'], 
                    pets[pets['OwnerID']==key]['Name'].to_list())
        else:
            return (owners.loc[key]['Name'],
                    pets[pets['OwnerID']==key]['Name'].to_list()[0])
        

    tuplist = pets['OwnerID'].apply(helper).to_list()
    idx, values = zip(*tuplist)
    out = pd.Series(values, idx)
    return out


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    """
    total cost per city

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_city(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['City'])
    True
    """
    procedure_history = procedure_history.merge(procedure_detail, on='ProcedureSubCode')[['PetID', 'ProcedureSubCode', 'Price']]

    procedure_history = procedure_history.groupby('PetID').sum()

    petsproc = pets.merge(procedure_history, on='PetID')

    petsproc = petsproc.merge(owners, on= 'OwnerID', how='left')[['City', 'Price']]

    out = petsproc.groupby('City').sum()['Price']
    return out

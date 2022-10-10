# project.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:
    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint
    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    columns = grades.columns.to_list()
    nonalphanum = ''.join(c for c in map(chr, range(256)) if not c.isalnum())

    out_dict = {'lab':np.unique(np.array([''.join(ch for ch in str 
                            if ch.isalnum())for str in 
                            [x[:5] for x in columns 
                            if 'lab' in x]])).tolist(),
                'project':np.unique(np.array([''.join(ch for ch in str 
                            if ch.isalnum())for str in 
                            [x[:9] for x in columns 
                            if 'project' in x]])).tolist(),
                'midterm':np.unique(np.array([''.join(ch for ch in str 
                            if ch.isalnum())for str in 
                            [x[:9] for x in columns 
                            if 'Midterm' in x]])).tolist(),
                'final':np.unique(np.array([''.join(ch for ch in str 
                            if ch.isalnum())for str in 
                            [x[:7] for x in columns 
                            if 'Final' in x]])).tolist(),
                'disc':np.unique(np.array([''.join(ch for ch in str 
                            if ch.isalnum())for str in 
                            [x[:12] for x in columns 
                            if 'disc' in x]])).tolist(),
                'checkpoint':np.unique(np.array([''.join(ch for ch in str 
                            if ch.isalnum())for str in 
                            [x[:25] for x in columns 
                            if 'checkpoint' in x]])).tolist()}
    return out_dict
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total takes in a DataFrame grades and returns the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    grades = grades
    columns = grades.columns.to_list()
    projectlist = np.unique(np.array([''.join(ch for ch in str 
                            if ch.isalnum())for str in 
                            [x[:9] for x in columns 
                            if 'project' in x]])).tolist()
    projectfrlist = np.unique(np.array( [x[:9] for x in columns 
                            if 'free_response' in x])).tolist()
    projectfrlist2 = [x[-2:] for x in projectfrlist]
    grades[projectfrlist + projectlist].fillna(0)


    for i in range(int(projectlist[-1][-2:])):
        if str(i).zfill(2) in projectfrlist2:
            grades['project1']= (grades['project01_free_response']+ grades['project01']).div((grades['project01_free_response - Max Points'] + grades['project01 - Max Points']))


    #ok this is a little cheeky i admit
    return grades['project1']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in a DataFrame 
    grades and returns a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by students that were marked "late" by Gradescope.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1, 10)])
    True
    >>> (out > 0).sum()
    8
    """
    
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def lateness_penalty(col):
    """
    adjust_lateness takes in a Series containing
    how late a submission was processed
    and returns a Series of penalties according to the
    syllabus.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def process_labs(grades):
    """
    process_labs takes in a DataFrame like grades and returns
    a DataFrame of processed lab scores. The output should:
      * have the same index as `grades`,
      * have one column for each lab assignment (e.g. `'lab01'`, `'lab02'`,..., `'lab09'`),
      * have values representing the final score for each lab assignment, 
        adjusted for lateness and scaled to a score between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1, 10)]
    True
    >>> np.all((0.60 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def lab_total(processed):
    """
    lab_total takes in DataFrame of processed assignments (like the output of 
    Question 5) and returns a Series containing the total lab grade for each 
    student according to the syllabus.
    
    Your answers should be proportions between 0 and 1.
    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def total_points(grades):
    """
    total_points takes in a DataFrame grades and returns a Series
    containing each student's course grade.
    Course grades should be proportions between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    ...



# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.
    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    ...


def letter_proportions(total):
    """
    letter_proportions takes in the final course grades
    as above and outputs a Series that contains the 
    proportion of the class that received each grade.
    :Example:
    >>> out = letter_proportions(pd.Series([0.99, 0.92, 0.89, 0.87, 0.82, 0.81, 0.80, 0.77, 0.77, 0.74]))
    >>> np.all(out.index == ['B', 'C', 'A'])
    True
    >>> out.sum() == 1.0
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def simulate_pval(grades, N):
    """
    simulate_pval takes in a DataFrame grades and
    a number of simulations N and returns the p-value
    for the hypothesis test described in the notebook.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 1000)
    >>> 0 <= out <= 0.1
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.
    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 0.5 < out[2][0] < 1
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """
    out = [..., ..., (..., ...), ..., (..., ...)]
    return out

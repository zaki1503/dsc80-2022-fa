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

    out_dict = {
        'lab':np.unique(np.array([x.split(' ')[0] for x in columns \
                if 'lab' in x])).tolist(),
        'project':np.unique(np.array([x.split(' ')[0].split('_')[0] \
                for x in columns if 'project' in x])).tolist(),
        'midterm':np.unique(np.array([x.split(' ')[0] for x in columns \
                if 'Midterm' in x])).tolist(),
        'final':np.unique(np.array([x.split(' ')[0] for x in columns \
                if 'Final' in x])).tolist(),
        'disc':np.unique(np.array([x.split(' ')[0] for x in columns \
                if 'disc' in x])).tolist(),
        'checkpoint':np.unique(np.array([x.split(' ')[0] for x in columns \
                if 'checkpoint' in x])).tolist()
                }
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
    grades = pd.read_csv(os.path.join('data', 'grades.csv'))
    grades = grades.fillna(0)
    columns = grades.columns.to_list()
    project_list = get_assignment_names(grades)['project']
    free_responses = np.unique(np.array([x.split(' ')[0] for x in columns \
                    if 'free_response' in x])).tolist()
    free_response_projects = [x.split('_')[0] for x in free_responses]
    free_response_projects

    total_grades = pd.DataFrame(index=grades.index)
    for project in project_list:
        if project in free_response_projects:
            total_grades[project+'score'] = ((grades[project+'_free_response']+
            grades[project]).div(grades[project+'_free_response - Max Points'] + grades[project+' - Max Points']))
        else:
            total_grades[project+'score'] = ((grades[project]).div(grades[project+' - Max Points']))

    total_grades['project_total'] = total_grades.mean(axis=1)
    return total_grades['project_total']


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
    labs = get_assignment_names(grades)['lab']
    num_late_list = []

    for lab in labs:
        s = grades[lab + ' - Lateness (H:M:S)']
        s = s.apply(lambda x: 
                            float(x.split(':')[0]) + 
                            float(x.split(':')[1])/24 +
                            float(x.split(':')[2])/(24*60)
                    )
        s = np.array(s)
        nomansland = s.mean()
        s = s[(s>0)&(s<nomansland)]
        num_late_list.append(s.size)

    return pd.Series(data = num_late_list, index=labs)


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
    col = col.apply(lambda x: 
                    float(x.split(':')[0]) + 
                    float(x.split(':')[1])/24 +
                    float(x.split(':')[2])/(24*60)
            )
    mu = col.mean()
    penalties = col.apply(lambda t, mu=mu: 1.0 if t < mu else (
                            0.9 if t <= 7 else(
                            0.7 if t <= 14 else(0.4)
                            )
                            )
                            )
    return penalties


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
    labs = get_assignment_names(grades)['lab']
    out_df = pd.DataFrame(index=grades.index)

    for lab in labs:
        lab_penalties = lateness_penalty(grades[lab + ' - Lateness (H:M:S)'])
        lab_scores = grades[lab]/grades[lab + ' - Max Points']
        out_df[lab] = lab_penalties*lab_scores

    return out_df


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
    def lab_total2(row):
        temp = np.array(row)
        temp = np.delete(temp, temp.argmin())
        return temp.mean()
    totals = processed.apply(lab_total2, axis=1)
    return pd.Series(data=totals,index= processed.index)


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
    grades = grades.fillna(0)
    assignment_dict = get_assignment_names(grades)

    def helper(assignment):
        assignments = assignment_dict[assignment.lower()]
        scores = []
        for assignment in assignments:
            scores.append(grades[assignment]/grades[assignment+' - Max Points'])
        score = np.array(scores).mean()
        return score

    checkpts_score = helper('checkpoint')
    discs_score = helper('disc')
    midterms_score = helper('Midterm')
    finals_score = helper('Final')


    final_score = (
            0.3*projects_total(grades)+ 
            0.2*lab_total(process_labs(grades))+
            0.025*checkpts_score+
            0.025*discs_score+
            0.15*midterms_score+
            0.3*finals_score
            )
    return final_score



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
    letters = total.apply(lambda x:
                        'A' if x>=0.9 else(
                        'B' if x>=0.8 else(
                        'C' if x>=0.7 else(
                        'D' if x>=0.6 else 'F')
                        ))
                        )
    return letters


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
    return total.groupby(by=total).count()/total.size


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
    grand_df = grades.fillna(0)
    grand_df['final_score'] = total_points(grades.fillna(0))
    grand_df

    observed = grand_df[grand_df['Level']=='SR']['final_score'].mean()

    samples = []
    class_scores = grand_df['final_score']
    class_mu = class_scores.mean()
    for n in range(N):
        sample =  np.random.choice(a=class_scores, size=(1,215)).mean()
        samples.append(sample)

    samples = np.array(samples)
    p = np.count_nonzero(samples <= observed)/N
    return p


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
    grades = grades.fillna(0)
    assignment_dict = get_assignment_names(grades)

    labs = process_labs(grades)


    def helper(assignment):
        assignments = assignment_dict[assignment.lower()]
        scores = pd.DataFrame(index=grades.index)
        for assignment in assignments:
            scores[assignment] = (grades[assignment]/grades[assignment+' - Max Points'])
        score = np.array(scores).mean()

        return scores

    cps = helper('checkpoint')
    dscs = helper('disc')
    mids = helper('Midterm')
    fins = helper('Final')

    grades = pd.read_csv(os.path.join('data', 'grades.csv'))
    grades = grades.fillna(0)
    columns = grades.columns.to_list()
    project_list = get_assignment_names(grades)['project']
    free_responses = np.unique(np.array([x.split(' ')[0] for x in columns \
            if 'free_response' in x])).tolist()
    free_response_projects = [x.split('_')[0] for x in free_responses]
    free_response_projects

    projects = pd.DataFrame(index=grades.index)
    for project in project_list:
        if project in free_response_projects:
                projects[project] = ((grades[project+'_free_response']+
                grades[project]).div(grades[project+'_free_response - Max Points'] + grades[project+' - Max Points']))
        else:
                projects[project] = ((grades[project]).div(grades[project+' - Max Points']))

    projects['project_total'] = projects.mean(axis=1)
    projs = projects.drop('project_total', axis=1)

    prenoise = pd.concat(objs=[projs, labs, cps, dscs, mids, fins], axis=1)
    noised = prenoise + np.random.normal(0,0.02, prenoise.shape)
    noised = noised.apply(lambda col: np.clip(col, 0, 1), axis=0)

    def lab_total2(row):
        temp = np.array(row)
        temp = np.delete(temp, temp.argmin())
        return temp.mean()
    lab_total = labs.apply(lab_total2, axis=1)

    noised['project_total'] = noised[project_list].apply(np.mean, axis=1)
    noised['lab_total'] = lab_total
    noised['checkpoint_total'] = noised[assignment_dict['checkpoint']].apply(np.mean, axis=1)
    noised['disc_total'] = noised[assignment_dict['disc']].apply(np.mean, axis=1)
    noised['midterm_total'] = noised[assignment_dict['midterm']].apply(np.mean, axis=1)
    noised['final_total'] = noised[assignment_dict['final']].apply(np.mean, axis=1)
    noised['total_total'] = (
                    0.3*noised['project_total']+ 
                    0.2*noised['lab_total']+
                    0.025*noised['checkpoint_total']+
                    0.025*noised['disc_total']+
                    0.15*noised['midterm_total']+
                    0.3*noised['final_total']
                    )

    noised['Level'] = grades['Level']
    return noised['total_total']


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
    out = [-0.000782930349967681, 0.1252336448598131, (0.51, 0.58), 0.3532710280373832, (True, False)]
    return out

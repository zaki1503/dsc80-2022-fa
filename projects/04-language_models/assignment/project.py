# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    time.sleep(6)
    try:
        r=requests.get(url)
        full= r.text
        truncate_start = full.split('*** START')[1]
        truncate_end = truncate_start.split("*** END")[0]

        remove_st = truncate_end.split('***')[1]
        remove_st = remove_st.replace("\r\n", "\n")

        return remove_st
    except Exception as e:
        print(e)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of every paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of every paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens should include no whitespace.
        - Whitespace (e.g. multiple newlines) between two paragraphs of text 
          should be ignored, i.e. they should not appear as tokens.
        - Two or more newlines count as a paragraph break.
        - All punctuation marks count as tokens, even if they are 
          uncommon (e.g. `'@'`, `'+'`, and `'%'` are all valid tokens).


    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    tokens = ["\x02"]

    test = book_string.strip()
    test = re.sub(pattern="\n{2,}", repl=" \x03 \x02 ", string= test)
    test = re.sub(pattern=" {1,}", repl=" ", string= test)
    test = re.sub(r"([\W]+|[^\w/'+$\s-]+)\s*", r" \1 ", string= test)
    test = re.sub(pattern="\n", repl=" ", string= test)
    test = re.sub(pattern="[.]", repl= " . ", string= test)

    tokens.extend(test.split())
    tokens.append("\x03")
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        out_dt = {token:1 for token in tokens}

        return pd.Series(out_dt)/len(out_dt)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        if all(i in self.mdl.index for i in set(words)):
            return np.prod([self.mdl[word] for word in words])
        else:
            return 0
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        lst = list(np.random.choice(
                            a = self.mdl.index, 
                            p= self.mdl.values, 
                            size=M
                            ))

        return ' '.join(lst)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        out_dt = {}

        for token in tokens:
            if token not in out_dt.keys():
                out_dt[token]= 1
            else:
                out_dt[token]+=1

        return pd.Series(out_dt)/len(tokens)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        if all(i in self.mdl.index for i in words):
            return np.prod([self.mdl[word] for word in words])
        else:
            return 0
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        lst = list(np.random.choice(
                            a = self.mdl.index, 
                            p= self.mdl.values, 
                            size=M))

        return ' '.join(lst)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        out = []
        for i in range(len(tokens)-self.N+1):
            out.append(tuple(tokens[i:i+self.N]))

        return out
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        # N-Gram counts C(w_1, ..., w_n)
        
        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        

        # Create the conditional probabilities
        
        
        # Put it all together
        out= pd.DataFrame(columns=["ngram", "n1gram", "prob"])

        ngrams = self.ngrams
        n1grams = [tuple(x[:-1]) for x in ngrams]
        out["ngram"] = ngrams
        out["n1gram"] = n1grams
        out["prob"] = [1 for x in ngrams]
        vcounts = out["n1gram"].value_counts()

        def helper(row):
            return row['prob']/vcounts[row['n1gram']]

        out["prob"] = out.apply(helper, axis=1)

        return out

    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4) * (1/2) * (1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        modelngrams, wordsngrams = list(self.mdl['ngram']), list(self.create_ngrams(words))

        prob = []

        if not all([ngram in modelngrams for ngram in wordsngrams]):
            return 0 #


        subgram = self.prev_mdl
        subgrams = []
        while hasattr(subgram, 'prev_mdl'):
            subgrams.append(subgram)
            subgram = subgram.prev_mdl

        subgrams.append(subgram)
        subgrams = subgrams[::-1]

        if not all([word in subgrams[0].mdl.index for word in words]):
            return 0 
            
        for i in range(len(subgrams)):
            if i == 0:
                prob.append(subgrams[i].mdl[words[0:i+1][0]])
                continue
            if tuple(words[0:i+1]) not in list(subgrams[i].mdl["ngram"]):
                return 0
            sdf = subgrams[i].mdl
            prob.append(float(sdf[sdf["ngram"] == tuple(words[0:i+1])]["prob"]))

        model = self.mdl
        for ngram in wordsngrams:
            prob.append(float(model[model["ngram"] == ngram]["prob"].sum()))

        return np.prod(prob)
        
    

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        string = ["\x02"]

        subgram = self.prev_mdl
        subgrams = []
        while hasattr(subgram, 'prev_mdl'):
            subgrams.append(subgram)
            subgram = subgram.prev_mdl

        subgrams.append(subgram)
        subgrams = subgrams[::-1][1:]

        for i in range(len(subgrams)):
            tdf = subgrams[0].mdl
            tdf = tdf[tdf["n1gram"]==tuple(string[i:i+1])]
            if tdf.shape[0] == 1:
                string.append(tdf['ngram'][0][-1])
            else:
                x = tdf

        i = len(subgrams)
        while i < M-1:
            tdf = self.mdl
            if tuple(string[i:i+self.N]) not in [tup for tup in tdf["n1gram"]]:
                for i in range(M-len(string)):
                    string.extend(["\x03"]*(M-len(string)))
                    break
            else:
                tdf = tdf[tdf["n1gram"]==tuple(string[i:i+self.N])]
                if tdf.shape[0] == 1:
                    string.append(tdf['ngram'].values[0][-1])
                else:
                    string.append(np.random.choice(
                                a=[ngram[-1] for ngram in tdf["ngram"]],
                                p=[p for p in tdf["prob"]]))
            
            i+=1

        string.append("\x03")
        return string
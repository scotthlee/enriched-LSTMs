import numpy as np
import pandas as pd

from sys import getsizeof
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer

# Generates bootstrap indices of a dataset with the option
# to stratify by one of the (binary-valued) variables
def boot_sample(df,
                size=None,
                by=None,
                pos_val=1,
                p=0.5,
                neg_replace=False,
                seed=None,
                return_indices=False):
    if seed == None:
        seed = np.random.randint(0, 1e6)
    if size == None:
        size = df.shape[0]
    if by == None:
        np.random.seed(seed)
        n = df.shape[0]
        sample_indices = np.random.choice(range(n),
                                          size=size,
                                          replace=True)
        if return_indices:
            out = sample_indices
        else:
            out = df.iloc[sample_indices, :]
    else:
        where_pos = np.where(df[by] == pos_val)[0]
        where_neg = np.where(df[by] != pos_val)[0]
        n_pos = int(p * size)
        n_neg = size - n_pos
        np.random.seed(seed)
        pos_indices = np.random.choice(where_pos,
                                       size=n_pos,
                                       replace=True)
        np.random.seed(seed)
        neg_indices = np.random.choice(where_neg,
                                       size=n_neg,
                                       replace=neg_replace)
        unshuff = np.concatenate([pos_indices, neg_indices])
        shuff = shuffle(unshuff, random_state=seed)
        if return_indices:
            out = shuff
        else:
            out = df.iloc[shuff, :]
    return out

# Returns only the unique words (or CCS codes) in a string;
# Borrowed from StackExchange coding forums
def keep_unique(visit):
	ucodes = []
	[ucodes.append(code) for code in visit.split() if code not in ucodes]
	return ' '.join(ucodes)

# Simple function for making one-hot vectors
def one_hot(indices, vocab_size, sparse=False, dtype=np.uint8):
    mat = np.zeros((vocab_size, indices.shape[0]), dtype=dtype)
    mat[indices, np.arange(mat.shape[1])] = 1
    mat[0, :] = 0
    return mat

# Converts a matrix of 1-hot vectors or probs to a single dense vector
def densify(mat, axis=2, flatten=False):
    out = np.argmax(mat, axis=axis)
    if flatten:
        out = out.flatten()
    return out

# Just a few functions to support making a sparse array
def sparsify(col, dtype='str', vocab_size=None, vocab_only=False):
    if dtype=='str':
        vec = CountVectorizer(binary=True,
                              ngram_range=(1, 1),
                              token_pattern="(?u)\\b\\w+\\b")
        data = vec.fit_transform(col)
        vocab = sorted(vec.vocabulary_.keys())
        if vocab_only:
            return vocab
    else:
        data = csr_matrix(tg.one_hot(col, vocab_size)).transpose()
        vocab = np.unique(col)
    return {'data':data, 'vocab':vocab}

# Calculates the proportion positive for a particular outcome
def prop_pos(label, outcomes):
    return np.true_divide(np.sum(outcomes == label), len(outcomes))

# Quick function for thresholding probabilities
def threshold(probs, cutoff=.5):
    return np.array(probs >= cutoff).astype(np.uint8)

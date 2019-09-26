import pandas as pd
import numpy as np
import h5py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz, save_npz, hstack, csr_matrix

import tools.generic as tg
import tools.text as tt

'''
Importing the data, renaming columns, and cleaning the simple text fields
'''
# Importing the data
filedir = 'C:/data/syndromic/'
records = pd.read_csv(filedir + 'nyc_2016_2017.txt',
                      encoding='latin1',
                      sep='\t')

# Renaming the columns
records.columns = [
    'admit_ct', 'hospcode', 'id', 'date',
    'time', 'age', 'gender', 'zip', 'admireason',
    'cc', 'icd', 'dd', 'disdate', 'travel', 'moa',
    'dispo', 'dischdt_ct', 'resp', 'ili', 'diar',
    'vomit', 'asthma'
]

# Dropping stuff we don't need
records = records.drop(['admit_ct', 'id', 'admireason', 'dd', 'travel',
                        'dischdt_ct', 'resp', 'ili', 'diar', 'vomit',
                        'asthma', 'zip'], axis=1)

# Cleaning the text fields; easy fixes first
records.hospcode = [doc.lower() for doc in records.hospcode]
records.age = [doc.replace('-', '') for doc in records.age]
records.moa = [doc.replace(' ', '').lower() for doc in records.moa]
records.gender = [doc.lower() for doc in records.gender.astype(str)]
records.dispo = [doc.replace(' ', '').lower()
                    for doc in records.dispo.astype(str)]

'''
Cleaning the ICD codes and converting them to CCS codes
'''
# Doing the ICD codes
icd = records.icd.astype(str)
icd = [doc.replace('.', '') for doc in icd]
icd = [doc.split('|') for doc in icd]
icd = [[code.replace(' ', '') for code in doc] for doc in icd]
icd = [[code.lower() for code in doc] for doc in icd]
records.icd = pd.Series([' '.join(codes) for codes in icd])

# Reading in the CCS lookup tables
ccs_icd9 = pd.read_csv(filedir + 'ccs_icd9.csv',
                       usecols=['ICD-9-CM CODE', 'CCS CATEGORY',
                                'CCS CATEGORY DESCRIPTION'])
ccs_icd10 = pd.read_csv(filedir + 'ccs_icd10.csv',
                        usecols=['ICD-10-CM CODE', 'CCS CATEGORY',
                                 'CCS CATEGORY DESCRIPTION'])
ccs_icd9.columns = ['icd', 'ccs', 'descr']
ccs_icd10.columns = ['icd', 'ccs', 'descr']
ccs_icd9.icd = [doc.lower() for doc in ccs_icd9.icd]
ccs_icd10.icd = [doc.lower() for doc in ccs_icd10.icd]
ccs_icd = pd.concat([ccs_icd9, ccs_icd10], axis=0)

# Making a lookup dictionary for the ICD codes
ccs_icd_dict = dict(zip([str(doc).lower() for doc in ccs_icd.icd],
                        [doc for doc in ccs_icd.ccs]))

# Keeping only codes that appear in the CCS lookup files
good_icd = [[code for code in doc if code in ccs_icd_dict.keys()]
            for doc in icd]

# Converting the ICD codes to CCS
ccs = [[str(ccs_icd_dict[code]) for code in doc] for doc in good_icd]
records['ccs'] = pd.Series([' '.join(list(set(doc))) for doc in ccs])

'''
Cleaning the chief complaints
'''
# Doing the bigger cleaning on chief complaint and discharge diagnosis
no_num_cc = pd.Series(
    tt.clean_column(
        records.cc.astype(str),
        remove_empty=False
        )
    )

num_cc = pd.Series(
    tt.clean_column(
        records.cc.astype(str),
        numerals=False,
        remove_empty=False
    )
)

records.cc = no_num_cc.astype(str)
records['num_cc'] = num_cc.astype(str)

# Weeding out records without CCs or ICD codes
records = records.iloc[np.where(records.cc != '')[0], :]
records = records.iloc[np.where(records.icd != 'nan')[0], :]
records = records.iloc[np.where(records.ccs != '')[0], :]
records = records.reset_index()

'''
Making the month, year, and hour columns, and exporting the clean records
'''

# Replacing 2016 with 16 so it matches (20)17
records.date = [doc.replace('2016', '16') for doc in records.date]

# Making a variable for hour
records['month'] = pd.Series([int(date[0:2]) for date in records.date],
                             dtype=np.uint8)
records['year'] = pd.Series([int(date[6:10]) for date in records.date],
                            dtype=np.uint8)
records['hour'] = pd.Series([int(time.split(':')[0]) for time in records.time],
                            dtype=np.uint8)

# Dropping date
records = records.drop(['date', 'time'], axis=1)

# Writing the file to disk
filedir = 'C:/data/syndromic/'
records.to_csv(filedir + 'records_unclipped.csv', index=False)

'''
Part 2: Converting CC to sequences of integers
'''
# Setting the text column to use for vectorization n stuff
text = [doc for doc in records.cc.astype(str)]

# First-pass vectorization to get the overall vocab
text_vec = CountVectorizer(binary=False,
                           ngram_range=(1, 1),
                           token_pattern="(?u)\\b\\w+\\b",
                           decode_error='ignore')
text_vec.fit(text)
vocab = text_vec.vocabulary_
vocab_size = len(list(vocab.keys()))

# Changing words with corpus counts < 5 to 'rareword'
doctermat = text_vec.transform(text)
word_sums = np.sum(doctermat, axis=0)
five_cols = np.where(word_sums < 5)[1]
where_five = np.where(np.sum(doctermat[:, five_cols], axis=1) > 0)[0]
for num in where_five:
    doc = text[num]
    for word in doc.split():
        if vocab[word] in five_cols:
            doc = doc.replace(word, 'rareword')
    text[num] = doc

# Writing the new text to disk
pd.Series(text).to_csv(filedir + 'unk5_cc.csv', index=False)

# Second-pass vectorization on the reduced-size corpus
five_vec = CountVectorizer(binary=False,
                           analyzer='word',
                           ngram_range=(1, 1),
                           token_pattern='\\b\\w+\\b',
                           decode_error='ignore')
five_vec.fit(text)
vocab = five_vec.vocabulary_
vocab_size = len(list(vocab.keys()))

# Adding 1 to each vocab index to allow for 0 masking
for word in vocab:
    vocab[word] += 1

# Writing the vocabulary to disk
vocab_df = pd.DataFrame.from_dict(vocab, orient='index')
vocab_df['word'] = vocab_df.index
vocab_df.columns = ['value', 'word']
vocab_df.to_csv(filedir + 'word_dict.csv', index=False)

# Converting the text strings to sequences of integers
# Clipping the docs to a max of 100 words
max_length = 100
clipped_docs = text.copy()
for i, doc in enumerate(text):
    if len(doc.split()) > max_length:
        clipped_docs[i] = ' '.join(doc.split()[0:max_length])

# Weeding out docs with tokens that CountVectorizer doesn't recognize;
# this shouldn't be necessary, but I can't figure out how to debug it.
in_vocab = np.where([np.all([word in vocab.keys()
                             for word in doc.split()])
                     for doc in clipped_docs])[0]
good_docs = [clipped_docs[i] for i in in_vocab]
good_recs = records.iloc[in_vocab, :]
good_recs.to_csv(filedir + 'records_clipped.csv', index=False)

# Preparing the HDF5 file to hold the output
output = h5py.File(filedir + 'word_sents.hdf5',  mode='w')

# Running and saving the splits for the inputs; going with np.uin16
# for the dtype since the vocab size is much smaller than before
int_sents = np.array([tt.pad_integers(tt.to_integer(doc.split(), vocab),
                                      max_length, 0) for doc in good_docs],
                     dtype=np.uint16)
output['sents'] = int_sents
output.close()

'''
Part 3: Converting the discrete variables to wide format
'''
# Reading in the data
slim_cols = ['age', 'gender', 'moa', 'hospcode', 'hour', 'year', 'month']
records = good_recs[slim_cols]

# Making the sparse matrices
sparse_out = [tg.sparsify(records[col].astype(str)) for col in slim_cols]
sparse_csr = hstack([col['data'] for col in sparse_out], format='csr')
sparse_vocab = [col['vocab'] for col in sparse_out]
sparse_vocab = pd.Series([item for sublist in sparse_vocab
                          for item in sublist])

# Writing the files to disk
save_npz(filedir + 'sparse_records', sparse_csr)
sparse_vocab.to_csv(filedir + 'sparse_vocab.csv', index=False)

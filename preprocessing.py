import pandas as pd
import numpy as np
import argparse
import h5py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz, save_npz, hstack, csr_matrix

import tools.generic as tg
import tools.text as tt

def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--clean_text', type=bool, default=False)
    parser.add_argument('--convert_numerals', type=bool, default=False)
    parser.add_argument('--min_df', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    '''
    Part 1: Reading in the data and processing the text
    '''
    
    # Importing the data
    records = pd.read_csv(args.data_dir + args.input_file)
    
    # Optional text cleaning
    if args.clean_text:
        text = tt.clean_column(records[args.text_column], 
                               numerals=args.convert_numerals)
    
    # Setting the text column to use for vectorization n stuff
    text = [doc for doc in records[args.text_column].astype(str)]
    
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
    lim_cols = np.where(word_sums < args.min_df)[1]
    where_lim = np.where(np.sum(doctermat[:, lim_cols], axis=1) > 0)[0]
    for num in where_lim:
        doc = text[num]
        for word in doc.split():
            if vocab[word] in lim_cols:
                doc = doc.replace(word, 'rareword')
        text[num] = doc
    
    # Writing the new text to disk
    pd.Series(text).to_csv(filedir + 'unk5_cc.csv', index=False)
    
    # Second-pass vectorization on the reduced-size corpus
    min_vec = CountVectorizer(binary=False,
                              nalyzer='word',
                              ngram_range=(1, 1),
                              token_pattern='\\b\\w+\\b',
                              decode_error='ignore')
    min_vec.fit(text)
    vocab = min_vec.vocabulary_
    vocab_size = len(list(vocab.keys()))
    
    # Adding 1 to each vocab index to allow for 0 masking
    for word in vocab:
        vocab[word] += 1
    
    # Writing the vocabulary to disk
    vocab_df = pd.DataFrame.from_dict(vocab, orient='index')
    vocab_df['word'] = vocab_df.index
    vocab_df.columns = ['value', 'word']
    vocab_df.to_csv(args.data_dir + 'word_dict.csv', index=False)
    
    # Converting the text strings to sequences of integers
    # Clipping the docs to a max of 100 words
    max_length = args.max_length
    clipped_docs = text.copy()
    for i, doc in enumerate(text):
        if len(doc.split()) > max_length:
            clipped_docs[i] = ' '.join(doc.split()[0:max_length])

    # Weeding out docs with tokens that CountVectorizer doesn't recognize;
    # this shouldn't be necessary, but I can't be bothered to debug it.
    in_vocab = np.where([np.all([word in vocab.keys()
                                 for word in doc.split()])
                         for doc in clipped_docs])[0]
    good_docs = [clipped_docs[i] for i in in_vocab]
    good_recs = records.iloc[in_vocab, :]
    good_recs.to_csv(args.data_dir + 'records_clipped.csv', index=False)

    # Preparing the HDF5 file to hold the output
    output = h5py.File(args.data_dir + 'word_sents.hdf5',  mode='w')

    # Running and saving the splits for the inputs; going with np.uin16
    # for the dtype since the vocab size is much smaller than before
    int_sents = np.array([tt.pad_integers(tt.to_integer(doc.split(), vocab),
                                          max_length, 0) for doc in good_docs],
                         dtype=np.uint16)
    output['sents'] = int_sents
    output.close()
    
    '''
    Part 2: Converting the discrete variables to wide format
    '''
    # Reading in the data
    slim_cols = list(records.columns.drop(args.text_column))
    records = good_recs[slim_cols]
    
    # Making the sparse matrices
    sparse_out = [tg.sparsify(records[col].astype(str)) for col in slim_cols]
    sparse_csr = hstack([col['data'] for col in sparse_out], format='csr')
    sparse_vocab = [col['vocab'] for col in sparse_out]
    sparse_vocab = pd.Series([item for sublist in sparse_vocab
                              for item in sublist])
    
    # Writing the files to disk
    save_npz(args.data_dir + 'sparse_records', sparse_csr)
    sparse_vocab.to_csv(args.data_dir + 'sparse_vocab.csv', index=False)

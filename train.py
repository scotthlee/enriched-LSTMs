import pandas as pd
import numpy as np
import h5py
import re
import os.path

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz

from models.supervised import RNN, EnrichedRNN

def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--target_column', type=str, default='code')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    # Importing the integer sequences
    int_sents = h5py.File(args.data_dir + 'word_sents.hdf5', mode='r')
    X = np.array(int_sents['sents'])
    int_sents.close()
    
    # Importing the sparse records and targets
    sparse_records = load_npz(args.data_dir + 'sparse_records.npz')
    records = pd.read_csv(args.data_dir + 'records_clipped.csv')
    
    # Making a variable for the classification target
    y = records[args.target_column]
    unique_codes = np.unique(y)
    if len(unique_codes) == 2:
        num_classes = 1
    else:
        num_classes = len(unique_codes)
    
    # Importing the vocabulary
    vocab_df = pd.read_csv(args.data_dir + 'word_dict.csv')
    vocab = dict(zip(vocab_df.word, vocab_df.value))
    
    # Defining the shared hyperparameters
    V = len(vocab)
    fit_batch = args.batch_size
    pred_batch = args.batch_size
    epochs = args.epochs
    max_length = X.shape[1]
    vocab_size = len(vocab.keys())
    seed = 10221983

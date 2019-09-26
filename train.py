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
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    # Directories for reading and writing data
    indir = args.data_dir
    outdir = args.data_dir
    
    # Importing the integer sequences
    int_sents = h5py.File(indir + 'word_sents.hdf5', mode='r')
    X = np.array(int_sents['sents'])
    int_sents.close()
    
    # Importing the sparse records and targets
    sparse_records = load_npz(indir + 'sparse_records.npz')
    records = pd.read_csv(indir + 'records_clipped.csv')
    
    # Defining some codes to use as targets
    target_codes = ['2', '88', '180', '233']
    code_regexes = [r'\b2\b', r'\b88\b', r'\b180\b', r'\b233\b']
    
    # Making logical variables from the regular expressions
    has = np.array([bool(re.search(hundo, doc))
                             for doc in records.num_cc], dtype=np.uint8)
    
    # Importing the vocabulary
    vocab_df = pd.read_csv(indir + 'word_dict.csv')
    vocab = dict(zip(vocab_df.word, vocab_df.value))
    
    # Defining the shared hyperparameters
    V = len(vocab)
    num_classes = 1
    fit_batch = args.batch_size
    pred_batch = args.batch_size
    epochs = args.epochs
    max_length = X.shape[1]
    vocab_size = len(vocab.keys())
    seed = 10221983

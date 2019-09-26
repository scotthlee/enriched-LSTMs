'''
This script compares the kbcENCE opioid overdose query to a text-only RNN
and a full-EHR encoder-decoder model across multiple train-test splits
'''
import pandas as pd
import numpy as np
import h5py
import re
import os.path

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz

import metrics.classification as mc
from models.supervised import RNN, EHRClassifier

# Directories for reading and writing data
indir = 'C:/data/syndromic/'
outdir = indir + 'ehr/'

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
fit_batch = 1024
pred_batch = 1024
epochs = 10
max_length = X.shape[1]
vocab_size = len(vocab.keys())
seed = 10221983

# Hyperparameters for the LSTM
rnn_e_size = 256
rnn_h_size = 256
rnn_e_drop = 0.75
rnn_r_drop = 0.25
rnn_f_drop = 0.00

# Hyperparameters for the multimodal model
sparse_size = 120
ehr_dense_size = 256
ehr_e_size = 256
ehr_e_drop = 0.85 
ehr_r_drop = 0.00

# Running the training loops
for i, code in enumerate(target_codes):
    # Setting the target
    y = np.array([bool(re.search(code_regexes[i], doc))
                             for doc in records.ccs], dtype=np.uint8)
    
    # Splitting the data
    train, not_train = train_test_split(list(range(X.shape[0])),
                                        test_size=.5,
                                        stratify=y,
                                        random_state=seed)
    val, test = train_test_split(not_train,
                                 test_size=.3,
                                 stratify=y[not_train],
                                 random_state=seed)
    
    # Training the LSTM
    rnn_modname = code + '_lstm'
    rnn_modpath = outdir + 'models/' + rnn_modname + '.hdf5'
    if os.path.isfile(rnn_modpath):
        rnn = load_model(rnn_modpath)
    else:
        rnn_check = ModelCheckpoint(filepath=rnn_modpath,
                                    save_best_only=True,
                                    verbose=1)
        rnn_stop = EarlyStopping(monitor='val_loss',
                                 patience=1)
        rnn = RNN(vocab_size=V,
                  output_size=num_classes,
                  embedding_size=rnn_e_size,
                  hidden_size=rnn_h_size,
                  embeddings_dropout=rnn_e_drop,
                  recurrent_dropout=rnn_r_drop,
                  final_dropout=rnn_f_drop,
                  cell_type='lstm')
        rnn.compile(optimizer='adam',
                      loss='binary_crossentropy')
        rnn.fit(X[train], y[train],
                  batch_size=fit_batch,
                  epochs=epochs,
                  validation_data=[X[val], y[val]],
                  callbacks=[rnn_check, rnn_stop])
        rnn = load_model(rnn_modpath)
    
    # Loading the trained rnn and getting the predictions
    rnn_val_preds = rnn.predict(X[val], batch_size=pred_batch)
    rnn_gm = mc.grid_metrics(y[val], rnn_val_preds)
    rnn_f1_cut = rnn_gm.cutoff[rnn_gm.f1.idxmax()]
    rnn_mcn_cut = rnn_gm.cutoff[np.argmin(abs(rnn_gm.rel))]
    rnn_test_preds = rnn.predict(X[test], batch_size=pred_batch).flatten()
    rnn_f1_guesses = mc.threshold(rnn_test_preds, rnn_f1_cut)
    rnn_mcn_guesses = mc.threshold(rnn_test_preds, rnn_mcn_cut)
    
    # Getting the summary statisics for the two cutoffs
    rnn_f1_stats = mc.binary_diagnostics(y[test], rnn_f1_guesses)
    rnn_mcn_stats = mc.binary_diagnostics(y[test], rnn_mcn_guesses)
    rnn_stats = pd.concat([rnn_f1_stats, rnn_mcn_stats], axis=0)
    rnn_stats['cutoff'] = np.array([rnn_f1_cut, rnn_mcn_cut]).reshape(-1, 1)
    
    # Writing the results to file
    rnn_stats.to_csv(outdir + 'stats/' + rnn_modname + '_stats.csv', 
                     index=False)
    
    for method in ['init', 'word', 'post']:
        # Training the encoder-decoder EHR model
        ehr_modname = code + '_' + method
        ehr_modpath = outdir + 'models/' + ehr_modname + '.hdf5'
        if os.path.isfile(ehr_modpath):
            ehr = load_model(ehr_modpath)
        else:
            ehr = EHRClassifier(sparse_size,
                                vocab_size,
                                max_length,
                                method=method,
                                output_size=1,
                                embedding_size=ehr_e_size,
                                hidden_size=ehr_dense_size,
                                embeddings_dropout=ehr_e_drop,
                                recurrent_dropout=ehr_r_drop)
            ehr_check = ModelCheckpoint(filepath=ehr_modpath,
                                        save_best_only=True,
                                        verbose=1)
            ehr_stop = EarlyStopping(monitor='val_loss',
                                     patience=1)
            ehr.compile(loss='binary_crossentropy',
                        optimizer='adam')
            ehr.fit([sparse_records[train], X[train]], y[train],
                    batch_size=fit_batch,
                    verbose=1,
                    epochs=epochs,
                    validation_data=([sparse_records[val], X[val]], y[val]),
                    callbacks=[ehr_check, ehr_stop])
            ehr = load_model(ehr_modpath)
        
        # Loading the trained EHR model and getting the predictions
        ehr_val_preds = ehr.predict([sparse_records[val], X[val]],
                                    batch_size=fit_batch)
        ehr_gm = mc.grid_metrics(y[val], ehr_val_preds)
        ehr_f1_cut = ehr_gm.cutoff[ehr_gm.f1.idxmax()]
        ehr_mcn_cut = ehr_gm.cutoff[np.argmin(abs(ehr_gm.rel))]
        ehr_test_preds = ehr.predict([sparse_records[test], X[test]],
                                     batch_size=fit_batch).flatten()
        ehr_f1_guesses = mc.threshold(ehr_test_preds, ehr_f1_cut)
        ehr_mcn_guesses = mc.threshold(ehr_test_preds, ehr_mcn_cut)
        
        # Getting the summary statistics for the EHR model
        ehr_f1_stats = mc.binary_diagnostics(y[test], ehr_f1_guesses)
        ehr_mcn_stats = mc.binary_diagnostics(y[test], ehr_mcn_guesses)
        ehr_stats = pd.concat([ehr_f1_stats, ehr_mcn_stats], axis=0)
        ehr_stats['cutoff'] = np.array([ehr_f1_cut, ehr_mcn_cut]).reshape(-1, 1)
        ehr_stats.to_csv(outdir + 'stats/' + ehr_modname + '_stats.csv', 
                         index=False)
        
    # Writing a file with the original records and each model's guesses
    '''
    test_recs = records.iloc[test, :].reindex()
    any_pos = np.array(y[test] + rnn_f1_guesses + rnn_mcn_guesses +
                       ehr_f1_guesses + ehr_mcn_guesses > 0, dtype=np.uint8)
    pos_recs = np.where(any_pos == 1)[0]
    possible_cases = np.where(any_pos == 1)[0]
    test_recs['rnn_f1'] = rnn_f1_guesses
    test_recs['rnn_mcn'] = rnn_mcn_guesses
    test_recs['ehr_f1'] = ehr_f1_guesses
    test_recs['ehr_mcn'] = ehr_mcn_guesses
    test_recs = test_recs.iloc[pos_recs, :]
    test_recs.to_csv(outdir + 'guesses/' + code + '.csv', index=False)
    '''

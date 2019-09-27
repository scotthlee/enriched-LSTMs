import pandas as pd
import numpy as np
import h5py
import re

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz

from models.supervised import RNN, EnrichedRNN

def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default=None,
                        help='the directory holding the data files')
    parser.add_argument('--text_file', type=str, default='word_sents.hdf5',
                        help='the HDF5 file holding the integer sentences')
    parser.add_argument('--records_npz', type=str, default='sparse_records.npz',
                        help='the NPZ file holding the sparsified variables')
    parser.add_argument('--records_csv', type=str, 
                        default='records_clipped.csv',
                        help='the CSV holding the original dataset')
    parser.add_argument('--target_column', type=str, default='code',
                        help='the target column in the original dataset')
    parser.add_argument('--train_batch_size', type=int, default=256,
                        help='minibatch size to use for training')
    parser.add_argument('--pred_batch_size', type=int, default=256,
                        help='minibatch size to use for inference')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to use for training')
    parser.add_argument('--train_size', type=float, default=0.5,
                        help='proportion of data to use for training')
    parser.add_argument('--seed', type=int, default=10221983,
                        help='seed to use for splitting the data')
    parser.add_argument('--enrichment_method', type=str, default='init',
                        choices=['word', 'init', 'post'],
                        help='method for encriching the LSTM')
    args = parser.parse_args()
    return args

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    # Importing the integer sequences
    int_sents = h5py.File(args.data_dir + args.integer_file, mode='r')
    X = np.array(int_sents['sents'])
    int_sents.close()
    
    # Importing the sparse records and targets
    sparse_records = load_npz(args.data_dir + args.records_npz)
    records = pd.read_csv(args.data_dir + args.records_csv)
    
    # Making a variable for the classification target
    y = records[args.target_column]
    unique_codes = np.unique(y)
    if len(unique_codes) == 2:
        num_classes = 1
        loss = 'binary_crossentropy'
    else:
        num_classes = len(unique_codes)
        loss = 'categorical_crossentropy'
    
    # Importing the vocabulary
    vocab_df = pd.read_csv(args.data_dir + 'word_dict.csv')
    vocab = dict(zip(vocab_df.word, vocab_df.value))
    
    # Defining the shared hyperparameters
    V = len(vocab)
    fit_batch = args.train_batch_size
    pred_batch = args.pred_batch_size
    epochs = args.epochs
    max_length = X.shape[1]
    vocab_size = len(vocab.keys())
    seed = args.seed
    
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
    
    # Setting the target
    y = args.records_cusv[args.target_column]
    
    # Splitting the data
    train, not_train = train_test_split(list(range(X.shape[0])),
                                        test_size=.5,
                                        stratify=y,
                                        random_state=seed)
    val, test = train_test_split(not_train,
                                 test_size=.3,
                                 stratify=y[not_train],
                                 random_state=seed)
    
    # Training the encoder-decoder EHR model
    ehr_modname = code + '_' + method
    ehr_modpath = args.data_dir + 'models/' + ehr_modname + '.hdf5'
    ehr = EHRClassifier(sparse_size,
                        vocab_size,
                        max_length,
                        method=method,
                        output_size=num_classes,
                        embedding_size=ehr_e_size,
                        hidden_size=ehr_dense_size,
                        embeddings_dropout=ehr_e_drop,
                        recurrent_dropout=ehr_r_drop)
    ehr_check = ModelCheckpoint(filepath=ehr_modpath,
                                save_best_only=True,
                                verbose=1)
    ehr_stop = EarlyStopping(monitor='val_loss',
                             patience=1)
    ehr.compile(loss=loss, optimizer='adam')
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
    ehr_stats.to_csv(args.data_dir + 'stats/' + ehr_modname + '_stats.csv', 
                     index=False)


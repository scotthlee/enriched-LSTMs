import numpy as np
import pandas as pd
import tensorflow as tf

from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import TimeDistributed, RepeatVector, Reshape
from keras.layers import concatenate, Multiply, Average, Softmax
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers import Embedding, LSTM, Bidirectional, GRU
from keras.layers import BatchNormalization, Dropout
from keras.layers import GlobalAveragePooling1D

# Basic RNN
def RNN(vocab_size,
        output_size,
        embedding_size=200,
        embeddings_dropout=0.2,
        hidden_size=100,
        recurrent_dropout=0.0,
        cell_type='lstm',
        final_dropout=0.0):
    
    # Building the embeddings layer
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_size,
                        mask_zero=True))
    
    # Specifying the type of RNN cell
    if cell_type == 'lstm':
        cell = LSTM(units=hidden_size,
                    dropout=embeddings_dropout,
                    recurrent_dropout=recurrent_dropout)
    elif cell_type == 'gru':
        cell = GRU(units=hidden_size,
                    dropout=embeddings_dropout,
                    recurrent_dropout=recurrent_dropout)
    else:
        raise Exception('cell_type should be either lstm or gru')
    model.add(cell)
    
    # Setting dropout and adding the Dense layer
    if final_dropout > 0:
        model.add(Dropout(final_dropout))
    if output_size == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    model.add(Dense(output_size, activation=activation))
    return model

# Hyrbid model for EHRs + clinical notes
def EnrichedLSTM(sparse_size,
                  vocab_size,
                  max_length,
                  method='init',
                  embedding_size=200,
                  embeddings_dropout=0.2,
                  hidden_size=100,
                  recurrent_dropout=0.0,
                  output_size=2,
                  trainable_records=True,
                  encoding_layer=None):
    # Reading the sparse version of the EHR variables
    input_record = Input(shape=(sparse_size,), name='ehr_input')
    
    # Embedding the EHR variables, optionally with pretrained weights
    if encoding_layer != None:
        ae_weights = encoding_layer.get_weights()
        record_embedding_layer = Dense(units=embedding_size,
                                       weights=ae_weights,
                                       trainable=trainable_records,
                                       name='ehr_embedding')
    else:
        record_embedding_layer = Dense(units=embedding_size,
                                       trainable=trainable_records,
                                       name='ehr_embedding')
    embedded_record = record_embedding_layer(input_record)
    
    # Building an embedding layer for the free text in the record
    input_text = Input(shape=(max_length,), name='text_input')
    embedding_layer = Embedding(input_dim=vocab_size,
                               output_dim=embedding_size,
                               mask_zero=True,
                               name='text_embedding')
    text_embedding = embedding_layer(input_text)
    
    # Setting the activation for the final layer
    if output_size == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    
    # Setting up the RNN
    rnn = LSTM(units=hidden_size,
               dropout=embeddings_dropout,
               recurrent_dropout=recurrent_dropout,
               return_sequences=False,
               return_state=True,
               name='rnn')
    
    # First option: pass the record as the initial state for the RNN
    if method == 'init':
        # Reshaping the record embedding
        reshaped_record = Reshape((1, embedding_size))(embedded_record)
        
        # Zero state for the RNN layer
        batch_size = K.shape(input_record)[0]
        zero_state = [K.zeros((batch_size, hidden_size)),
                      K.zeros((batch_size, hidden_size))]
        
        # Running the record through the RNN first, and then the text
        rec_out, rec_h, rec_c = rnn(reshaped_record,
                                    initial_state=zero_state)
        pre_dense, _, _ = rnn(text_embedding,
                              initial_state=[rec_h, rec_c])
    
    # Second option: concat the RNN output and the record before softmax
    elif method == 'post':
        rnn_output, _, _ = rnn(text_embedding)
        pre_dense = concatenate([embedded_record, rnn_output])
    
    # Third option: concat the word embeddings and the (repeated) record emb.
    elif method == 'word':
        repeated_record = RepeatVector(max_length)(embedded_record)
        text_embedding = concatenate([text_embedding, repeated_record], 2)
        pre_dense, _, _ = rnn(text_embedding)
    
    # Adding the final dense layer
    print(pre_dense.shape)
    output = Dense(units=output_size, activation=activation)(pre_dense)
    
    # Putting everything together
    model = Model([input_record, input_text], output)
    return model

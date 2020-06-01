#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.layers import Input, Dense, Bidirectional, LSTM, Lambda, concatenate, average
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam

import numpy as np
import random
import pickle

from datetime import datetime
import sys
import logging
import os


'''
Reused sequential
'''
def seq_lstm():
    lstm = Sequential()
    lstm.add(LSTM(200, return_sequences=True))
    lstm.add(LSTM(100))
    lstm.add(Dense(48,activation='relu'))
    return lstm

def seq_representation():
    repres = Sequential()
    repres.add(Dense(64,activation='relu'))
    repres.add(Dense(32,activation='relu'))
    repres.add(Dense(8,activation='relu'))
    return repres

def seq_profile():
    # model profile features representation nn
    profile = Sequential()
    profile.add(Dense(64,activation='relu'))
    profile.add(Dense(32,activation='relu'))
    profile.add(Dense(8,activation='relu'))
    return profile

def seq_similarity():
    prediction = Sequential()
    prediction.add(Dense(64,activation='relu'))
    prediction.add(Dense(32,activation='relu'))
    prediction.add(Dense(8,activation='relu'))
    prediction.add(Dense(1,activation='sigmoid'))
    return prediction


'''
Build ST-SiameseNet and other compared models
'''
def build_model_best(with_speed,with_profile):
    if with_speed:
        inputs1_d1s = [Input((None,4)) for _ in range(5)] 
        inputs1_d1d = [Input((None,4)) for _ in range(5)] 
        inputs1_d2s = [Input((None,4)) for _ in range(5)] 
        inputs1_d2d = [Input((None,4)) for _ in range(5)] 
    else:
        inputs1_d1s = [Input((None,3)) for _ in range(5)] 
        inputs1_d1d = [Input((None,3)) for _ in range(5)] 
        inputs1_d2s = [Input((None,3)) for _ in range(5)] 
        inputs1_d2d = [Input((None,3)) for _ in range(5)] 

    # build up model
    # model two LSTM1
    seq_lstm1 = seq_lstm()
    # model two LSTM2
    seq_lstm2 = seq_lstm()
    # model representation nn
    seq_repres = seq_representation()
    # similarity nn
    seq_sim = seq_similarity()

    # input to lstm
    lstm1_d1s = [seq_lstm1(traj_input) for traj_input in inputs1_d1s]
    lstm1_d1d = [seq_lstm2(traj_input) for traj_input in inputs1_d1d]
    lstm1_d2s = [seq_lstm1(traj_input) for traj_input in inputs1_d2s]
    lstm1_d2d = [seq_lstm2(traj_input) for traj_input in inputs1_d2d]

    # get trip embeddings
    trip_emb_d1 = concatenate(lstm1_d1s+lstm1_d1d)
    trip_emb_d2 = concatenate(lstm1_d2s+lstm1_d2d)

    # one day one driver has one profiel features
    if with_profile:
        # inputs 2: profile feature
        inputs2_d1 = Input((11,)) 
        inputs2_d2 = Input((11,)) 
        # model profile features representation nn
        seq_pro = seq_profile()
        # get profile embeddings
        pro_emb_d1 = seq_pro(inputs2_d1)
        pro_emb_d2 = seq_pro(inputs2_d2)

        # concatenate xyt(v) and profile 
        cat = concatenate([trip_emb_d1]+[pro_emb_d1]+[trip_emb_d2]+[pro_emb_d2])
    else:
        cat = concatenate([trip_emb_d1]+[trip_emb_d2])
    # merge input and output
    inputs_tmp = inputs1_d1s+inputs1_d1d+inputs1_d2s+inputs1_d2d
    if with_profile:
        inputs_tmp.append(inputs2_d1)
        inputs_tmp.append(inputs2_d2)
        
    # similarity nn for learning xyt and profile together
    prediction = seq_sim(cat)
    
    # training process
    siamese_net = Model(inputs=inputs_tmp,outputs=prediction)
    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    return siamese_net


def build_model_profileonly():
    # define inputs
    inputs2_d1 = Input((11,)) 
    inputs2_d2 = Input((11,)) 
    
    # model profile features representation nn
    seq_pro = seq_profile()
    # similarity nn 
    seq_sim = seq_similarity()
    
    # get profile embeddings
    pro_emb_d1 = seq_pro(inputs2_d1)
    pro_emb_d2 = seq_pro(inputs2_d2)
    # concatenate embeddings
    cat = concatenate([pro_emb_d1]+[pro_emb_d2])
    # get prediction
    # similarity nn for learning xyt and profile together
    prediction = seq_sim(cat)
        
    siamese_net = Model(inputs=[inputs2_d1,inputs2_d2],outputs=prediction)
    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    
    return siamese_net

def build_model_seekserve(with_speed,with_profile):
    if with_speed: 
        inputs_d1 = [Input((None,4)) for _ in range(5)] 
        inputs_d2 = [Input((None,4)) for _ in range(5)] 
    else: 
        inputs_d1 = [Input((None,3)) for _ in range(5)] 
        inputs_d2 = [Input((None,3)) for _ in range(5)] 
    
    # model lstm
    seq_lstm1 = seq_lstm()
    # representation nn
    seq_repres = seq_representation()
    # similarity nn
    seq_sim = seq_similarity()
    
    # input to lstm
    lstm_d1 = [seq_lstm1(traj_input) for traj_input in inputs_d1]
    lstm_d2 = [seq_lstm1(traj_input) for traj_input in inputs_d2]
    
    # get trip embeddings
    temp_emb_d1 = concatenate(lstm_d1)
    temp_emb_d2 = concatenate(lstm_d2)
    
    # one day one driver has one profiel features
    if with_profile:
        # inputs 2: profile feature
        inputs2_d1 = Input((11,)) 
        inputs2_d2 = Input((11,)) 
        # model profile features representation nn
        seq_pro = seq_profile()
        # get profile embeddings
        pro_emb_d1 = seq_pro(inputs2_d1)
        pro_emb_d2 = seq_pro(inputs2_d2)
        # concatenate xyt(v) and profile 
        cat = concatenate([temp_emb_d1]+[pro_emb_d1]+[temp_emb_d2]+[pro_emb_d2])
    else:
        cat = concatenate([temp_emb_d1]+[temp_emb_d2])
    
    # merge input and output
    inputs_tmp = inputs_d1+inputs_d2
    if with_profile:
        inputs_tmp.append(inputs2_d1)
        inputs_tmp.append(inputs2_d2)
        
    # similarity nn or nn for learning xyt and profile together
    prediction = seq_sim(cat)
    
    siamese_net = Model(inputs=inputs_tmp,outputs=prediction)
    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    return siamese_net

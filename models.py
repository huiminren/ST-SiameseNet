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



def seq_lstm():
    lstm = Sequential()
    lstm.add(LSTM(200, return_sequences=True))
    lstm.add(LSTM(100))
    lstm.add(Dense(48,activation='relu'))
    return lstm


# In[3]:


def seq_representation():
    repres = Sequential()
    repres.add(Dense(64,activation='relu'))
    repres.add(Dense(32,activation='relu'))
    repres.add(Dense(8,activation='relu'))
    return repres


# In[4]:


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

# In[6]:

def build_model_best(with_speed,with_profile,with_temporal,nn_sim_flag):
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

    if with_temporal:
        # get temporal embeddings
        temp_emb_d1 = seq_repres(trip_emb_d1)
        temp_emb_d2 = seq_repres(trip_emb_d2)
    else:
        temp_emb_d1 = trip_emb_d1
        temp_emb_d2 = trip_emb_d2

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
        if nn_sim_flag:
            cat = concatenate([temp_emb_d1]+[pro_emb_d1]+[temp_emb_d2]+[pro_emb_d2])
        else:
            rep1 = concatenate([temp_emb_d1]+[pro_emb_d1])
            rep2 = concatenate([temp_emb_d2]+[pro_emb_d2])
    else:
        if nn_sim_flag:
            cat = concatenate([temp_emb_d1]+[temp_emb_d2])
        else:
            rep1 = temp_emb_d1
            rep2 = temp_emb_d2
    # merge input and output
    inputs_tmp = inputs1_d1s+inputs1_d1d+inputs1_d2s+inputs1_d2d
    if with_profile:
        inputs_tmp.append(inputs2_d1)
        inputs_tmp.append(inputs2_d2)
        
    # similarity nn or nn for learning xyt and profile together
    if nn_sim_flag:
        prediction = seq_sim(cat)
    else:
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        #call this layer on list of two input tensors.
        L1_distance = L1_layer([rep1, rep2])
        prediction = Dense(1,activation='sigmoid')(L1_distance)

    siamese_net = Model(inputs=inputs_tmp,outputs=prediction)
    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    return siamese_net


# In[18]:


def build_model_profileonly(nn_sim_flag):
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
    if nn_sim_flag:
        # similarity nn or nn for learning xyt and profile together
        prediction = seq_sim(cat)
    else:
        # l1 similarity
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        #call this layer on list of two input tensors.
        L1_distance = L1_layer([pro_emb_d1, pro_emb_d2])
        prediction = Dense(1,activation='sigmoid')(L1_distance)
        
    siamese_net = Model(inputs=[inputs2_d1,inputs2_d2],outputs=prediction)
    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    
    return siamese_net


# In[27]:


def build_model_seekserve(with_speed,with_profile,with_temporal,nn_sim_flag):
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
    emb_d1 = concatenate(lstm_d1)
    emb_d2 = concatenate(lstm_d2)
    
    if with_temporal:
        # get temporal embeddings
        temp_emb_d1 = seq_repres(emb_d1)
        temp_emb_d2 = seq_repres(emb_d2)
    else:
        temp_emb_d1 = emb_d1
        temp_emb_d2 = emb_d2
    
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
        if nn_sim_flag:
            cat = concatenate([temp_emb_d1]+[pro_emb_d1]+[temp_emb_d2]+[pro_emb_d2])
        else:
            rep1 = concatenate([temp_emb_d1]+[pro_emb_d1])
            rep2 = concatenate([temp_emb_d2]+[pro_emb_d2])
    else:
        if nn_sim_flag:
            cat = concatenate([temp_emb_d1]+[temp_emb_d2])
        else:
            rep1 = temp_emb_d1
            rep2 = temp_emb_d2
    
    # merge input and output
    inputs_tmp = inputs_d1+inputs_d2
    if with_profile:
        inputs_tmp.append(inputs2_d1)
        inputs_tmp.append(inputs2_d2)
    # prediction layer
    if nn_sim_flag:
        # similarity nn or nn for learning xyt and profile together
        prediction = seq_sim(cat)
    else:
        # l1 similarity
        #layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        #call this layer on list of two input tensors.
        L1_distance = L1_layer([rep1, rep2])
        prediction = Dense(1,activation='sigmoid')(L1_distance)
    
    siamese_net = Model(inputs=inputs_tmp,outputs=prediction)
    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    return siamese_net

#!/usr/bin/env python
# coding: utf-8

import pickle
import random
import numpy as np

def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def load_data(path,filename):
    '''
    Args:
    -----
        path: path
        filename: file name
    Returns:
    --------
        data: loaded data
    '''
    file = open(path+filename, 'rb')
    data = pickle.load(file)
    file.close() 
    return data

def expand(data):
    '''
    Expand dimension for data
    traj.shape == (num_states,num_features) ==> traj.shape == (1,num_states,num_features)
    '''
    data = np.expand_dims(data, axis=0)
    return data 

def get_days(ispos = True, num_days = 5,new_days = False):
    '''
    Return two days for positive or negative pair
    Args:
    ----
    ispos: boolean
        True: positive 
        False: negative
    num_days: int
        1 to 9
    new_days: boolean
        True: new days
        False: old days
    Return:
    -------
    
    '''
    if ispos:
        if not new_days:
            pos_rnd_days = random.sample(range(num_days),2) # randomly pick up 2 days from the first num_days days. 
            return pos_rnd_days
        else:
            if num_days == 9:
                pos_rnd_days = [8,9]
                return pos_rnd_days
            else:
                pos_rnd_days = random.sample(range(num_days,10),2) # randomly pick up 2 days from the rest days (new).
                return pos_rnd_days
    else:
        neg_rnd_days = []
        if not new_days:
            for _ in range(2):
                neg_rnd_days.extend(random.sample(range(num_days),1)) #  randomly pick up 1 day twice from first num_days days. 
            return neg_rnd_days
        else:
            for _ in range(2):
                neg_rnd_days.extend(random.sample(range(num_days,10),1)) #  randomly pick up 1 day twice from the rest days(new). 
            return neg_rnd_days

def get_trajs(data, rnd_plt, rnd_day, input_type):
    '''
    Args:
    -----
    data: list of trajs (xyt or xytv)
    rnd_plt: str
    rnd_day: int
    input_type: 'seek' or 'serve'
    Return:
    -------
    trajs: 5 traj in the list
    '''
    # find trajs of the plate and the day and the type
    trajs = data[rnd_plt][rnd_day][input_type]
    # randomly select 5 trajectories for each day
    trajs = random.sample(trajs,5)
    return trajs

def merge_data(t,d1_profile,d2_profile):
    """
    merge all data, final step for preparing pair
    Args:
    ------
    t: list. [] or xyt(v) list
    d1_profile: list
    d2_profile: list
    Return:
    -------
    inputs: merged data
    """
    t.append(d1_profile)
    t.append(d2_profile)
    inputs = np.array(t)
    inputs = [expand(data) for data in inputs]
    return inputs
    
def get_pairs_s_or_d(data, profile_data, plates, input_type, num_days = 5, new_days = False):
    '''
    Get input pairs and label, batch_size = 1
    For postive pair,
        randomly select one driver and two days
        randomly select 5 seeking or 5 driving
    For negtive pair,
        randomly select two drivers and two days
        randomly select 5 seeking or 5 driving
    
    seek: xyt+(v)+(profile)
    drive: xyt+(v)+(profile)
    
    Args:
    -----
    data:
        xyt+v data
    profile_data:
        11 dimension profile features, one vector for one day one driver
    plates:
        plates containing enough data
    input_type:
        'seek' or 'serve'
    num_days:
        number of days to train
    new_days:
        flag, if use new days to test or validation
    Return:
    -------
        randomly return an input and lable, either positive or negative pair and label
        (input,label)
    '''
    # 0.5 probability to return positive pair
    if random.random()<=0.5: 
        # postive pair
        pos_rnd_days = get_days(ispos = True, num_days = num_days, new_days = new_days)
        pos_rnd_plt = random.sample(plates,1) # randomly pick up one driver
        
        d1 = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[0], input_type)
        d2 = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[1], input_type)
        
        if len(profile_data)>0: # if using profile data, add to the pair
            d1_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[0]]
            d2_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[1]]
            t_pos = d1+d2
            inputs_pos = merge_data(t_pos,d1_profile_pos,d2_profile_pos)
            
        else: # otherwise only using xyt(v) trajs as pair
            t_pos = d1+d2
            inputs_pos = [expand(data) for data in np.array(t_pos)]
            
        return inputs_pos,[0]
    
    else:
        # negative pair
        neg_rnd_days = get_days(ispos = False, num_days = num_days, new_days = new_days)
        neg_rnd_plt = random.sample(plates,2) # randomly pick up two drivers
        
        nd1 = get_trajs(data, neg_rnd_plt[0], neg_rnd_days[0], input_type)
        nd2 = get_trajs(data, neg_rnd_plt[1], neg_rnd_days[1], input_type)
        
        if len(profile_data)>0:
            d1_profile_neg = profile_data[neg_rnd_plt[0]][neg_rnd_days[0]]
            d2_profile_neg = profile_data[neg_rnd_plt[1]][neg_rnd_days[1]]
            t_neg = nd1+nd2
            inputs_neg = merge_data(t_neg,d1_profile_neg,d2_profile_neg)
        else:
            t_neg = nd1+nd2
            inputs_neg = [expand(data) for data in np.array(t_neg)]
            
        return inputs_neg,[1]

def get_pairs_s_and_d(data, profile_data, plates, input_type='', num_days = 5, new_days = False):
    '''
    Get input pairs and label, batch_size = 1
    For postive pair,
        randomly select one driver and two days
        randomly select 5 seeking and 5 driving
    For negtive pair,
        randomly select two drivers and two days
        randomly select 5 seeking and 5 driving
    
    seek+drive: xyt
    seek+drive: xyt+v
    seek+drive: xyt+v+profile
    seek+drive: profile
    
    using plates and new_days to return validation and test dataset
    
    Args:
    -----
    data:
        xyt+(v) data
    profile_data:
        11 dimension profile features, one vector for one day one driver
    plates:
        plates containing enough data
    input_type:
        keep consistant with get_pairs_s_or_d, furture used in _acc
    num_days:
        number of days to train
    new_days:
        flag, if use new days to test or validation
    Return:
    -------
        randomly return an input and lable, either positive or negative pair and label
        (input,label)
    '''
    # 0.5 probability to return positive pair
    if random.random()<=0.5: 
        # postive pair
        pos_rnd_days = get_days(ispos = True, num_days = num_days, new_days = new_days)
        pos_rnd_plt = random.sample(plates,1) # randomly pick up one driver
        if len(data)>0:# if using trajs data
            d1s = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[0], 'seek')
            d1d = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[0], 'serve')
            d2s = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[1], 'seek')
            d2d = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[1], 'serve')

            if len(profile_data)>0: # if using profile data, add to the pair
                d1_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[0]]
                d2_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[1]]
                t_pos = d1s + d1d + d2s + d2d
                inputs_pos = merge_data(t_pos,d1_profile_pos,d2_profile_pos)

            else: # otherwise only using xyt(v) trajs as pair
                t_pos = d1s + d1d + d2s + d2d
                inputs_pos = [expand(data) for data in np.array(t_pos)]
                
        else: # if only using profile data
            
            d1_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[0]]
            d2_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[1]]
            inputs_pos = merge_data([],d1_profile_pos,d2_profile_pos)
            
        return inputs_pos,[0]
    
    else:
        # negative pair
        neg_rnd_days = get_days(ispos = False, num_days = num_days, new_days = new_days)
        neg_rnd_plt = random.sample(plates,2) # randomly pick up two drivers
        if len(data)>0:
            nd1s = get_trajs(data, neg_rnd_plt[0], neg_rnd_days[0], 'seek')
            nd1d = get_trajs(data, neg_rnd_plt[0], neg_rnd_days[0], 'serve')
            nd2s = get_trajs(data, neg_rnd_plt[1], neg_rnd_days[1], 'seek')
            nd2d = get_trajs(data, neg_rnd_plt[1], neg_rnd_days[1], 'serve')

            if len(profile_data)>0:
                d1_profile_neg = profile_data[neg_rnd_plt[0]][neg_rnd_days[0]]
                d2_profile_neg = profile_data[neg_rnd_plt[1]][neg_rnd_days[1]]
                t_neg = nd1s + nd1d + nd2s + nd2d
                inputs_neg = merge_data(t_neg,d1_profile_neg,d2_profile_neg)
            else:
                t_neg = nd1s + nd1d + nd2s + nd2d
                inputs_neg = [expand(data) for data in np.array(t_neg)]
        else:
            d1_profile_neg = profile_data[neg_rnd_plt[0]][neg_rnd_days[0]]
            d2_profile_neg = profile_data[neg_rnd_plt[1]][neg_rnd_days[1]]
            inputs_neg = merge_data([],d1_profile_neg,d2_profile_neg)
            
        return inputs_neg,[1]

def save_model(net, model_path, tag):
    # serialize model to JSON
    model_json = net.to_json()
    with open(model_path+"model_{0}.json".format(tag), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    net.save_weights(model_path+"model_{0}.h5".format(tag))
    print("Saved model to disk")

def acc(net, pairs, labels):
    '''
    net: 
        trained network
    pairs:
    labels:
    '''
    n_correct = 0
    prob_list = []
    for i in range(len(pairs)): # evaluate 1000 pairs
        prob = net.predict(pairs[i])
        prob_list.append(prob[0][0])
        if 1*(prob[0][0]>0.5) == labels[i][0]:
            n_correct+=1
    acc = n_correct/len(labels)
    return acc,prob_list
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils import *
from models import *
import os
import argparse

def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(description='ST-SiameseNet')
    # model hyper-parameters
    parser.add_argument('--with_speed', action='store_true', 
                        help='input trajs with speed')
    parser.add_argument('--with_profile', action='store_true', 
                        help='input trajs with profile features')
    parser.add_argument('--input_type', type=str, default='all', 
                        help='input type, seek, serve or all')
    parser.add_argument('--num_train_plates', type=int, default=500,
                        help='number of training plates (default: 500)')
    parser.add_argument('--num_days', type=int, default=5,
                        help='number of training days (default: 5)')
    
    # training hyper-parameters
    parser.add_argument('--iteration', type=int, default=1000000, 
                        help='number of iterations (default: 1000000)')
    
    # saving and loading directoreis
    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--log_step', type=int , default=1000)
    parser.add_argument('--checkpoint_every', type=int , default=50000)
    
    return parser
    
def main(opts):
    """
    Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """
    create_dir(opts.log_path)
    create_dir(opts.model_path)

    # prepare logging file
    tag = str(opts.num_train_plates)+'plates_'+'days'+str(opts.num_days)+\
    '_inputs_'+str(opts.input_type)+'_speed'+str(opts.with_speed)+'_profile'+str(opts.with_profile)
    print(tag)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=log_path+tag+'.log',
                        filemode='a')

    # load data
    all_plates = load_data(opts.data_path,'plates.pkl')
    train_plates = all_plates[:opts.num_train_plates]
    test_plates = all_plates[2000:] # 197 plates which are unseen

    if opts.input_type == 'all':
        input_type = ''
    else:
        input_type = opts.input_type
        
    if opts.with_speed:
        raw_trajs = load_data(opts.data_path,'trajs_with_speed.pkl')
    else:
        raw_trajs = load_data(opts.data_path,'trajs_without_speed.pkl')
        
    if opts.with_profile:
        profile_data = load_data(opts.data_path,'profile_features.pkl')
    else:
        profile_data = []

    # prepare model and input data
    if (input_type == '') and (with_speed == False) and (with_profile==True):
        # profile is True and speed is False --> profile only (no xyt)
        siamese_net = build_model_profileonly()
        get_pairs = get_pairs_s_and_d
        raw_trajs = []
    elif input_type == '':
        # both speed and profile are True --> xyt+v+profile
        # both speed and profile are False --> xyt
        # speed is True and profie is False --> xyt+v
        siamese_net = build_model_best(with_speed,with_profile)
        get_pairs = get_pairs_s_and_d
    elif input_type != '':
        # when input_type != '' --> seek or drive 
        siamese_net = build_model_seekserve(with_speed,with_profile)
        get_pairs = get_pairs_s_or_d

    # start training 
    loss_500 = []
    train_acc_500 = []
    val_acc_500 = []
    test_acc_500 = []

    iteration = opts.iteration
    # save best model
    current_train_acc = 0.7
    current_val_acc = 0.7
    current_test_acc = 0.7
    t0 = datetime.now()
    t1 = datetime.now()

    # prepare evaluation dataset
    pairs,labels = [],[]
    val_pairs,val_labels = [],[]
    test_pairs,test_labels = [],[] 
    for _ in range(1000):
        pv,lv = get_pairs(raw_trajs, profile_data, train_plates, input_type, num_days,new_days = True)
        val_pairs.append(pv) 
        val_labels.append(lv)
        pt,lt = get_pairs(raw_trajs, profile_data, test_plates, input_type, num_days,new_days = True)
        test_pairs.append(pt) 
        test_labels.append(lt)

    for ite in range(iteration):
        pair,label = get_pairs(raw_trajs, profile_data, train_plates, input_type, num_days)
        loss = siamese_net.train_on_batch(pair,label)
        pairs.append(pair)
        labels.append(label)
        # save log
        if ite % opts.log_step == 0 and ite != 0:
            t1 = datetime.now()
            print(ite)
            train_acc,train_prob_list = acc(siamese_net, pairs, labels)
            pairs = []
            labels = []
            val_acc, val_prob_list = acc(siamese_net, val_pairs, val_labels) # old plates new days
            test_acc,test_prob_list = acc(siamese_net, test_pairs, test_labels) # test with new plates and new days.
            loss_500.append(loss)
            train_acc_500.append(train_acc)
            val_acc_500.append(val_acc)
            test_acc_500.append(test_acc)
            logging.info('******iteration: '+str(ite)+'; loss: '+str(loss)+ '; train acc: '+str(train_acc)+'; validation acc: '+ str(val_acc)+'; test acc: '+str(test_acc))
            if train_acc > current_train_acc:
                save_model(siamese_net, model_path, tag = tag+'_best_train')
                current_train_acc = train_acc
                logging.info('best train model updated: ' + str(train_acc))
            if val_acc > current_val_acc:
                save_model(siamese_net, model_path, tag = tag+'_best_val')
                current_val_acc = val_acc
                logging.info('best validation model updated: ' + str(val_acc))
            if test_acc > current_test_acc:
                save_model(siamese_net, model_path, tag = tag+'_best_test')
                current_test_acc = test_acc
                logging.info('best test model updated: ' + str(test_acc))
        if ite % opts.checkpoint_every == 0:
            save_model(siamese_net, model_path, tag = tag + '_mid')
            pickle.dump(loss_500, open(model_path+'loss_{0}.pkl'.format(tag),'wb'))
            pickle.dump(train_acc_500, open(model_path+'train_acc_{0}.pkl'.format(tag),'wb'))
            pickle.dump(val_acc_500, open(model_path+'val_acc_{0}.pkl'.format(tag),'wb'))
            pickle.dump(test_acc_500, open(model_path+'test_acc_{0}.pkl'.format(tag),'wb'))
    logging.info('total running time: '+ str(datetime.now()-t0))
    save_model(siamese_net, model_path, tag = tag + '_iter'+str(iteration))
    pickle.dump(loss_500, open(model_path+'loss_{0}.pkl'.format(tag),'wb'))
    pickle.dump(train_acc_500, open(model_path+'train_acc_{0}.pkl'.format(tag),'wb'))
    pickle.dump(val_acc_500, open(model_path+'val_acc_{0}.pkl'.format(tag),'wb'))
    pickle.dump(test_acc_500, open(model_path+'test_acc_{0}.pkl'.format(tag),'wb'))

if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    
    print(opts)
    main(opts)

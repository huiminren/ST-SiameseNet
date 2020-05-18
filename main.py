#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils import *
from models import *
import os


log_path = "./log/"
if not os.path.isdir(log_path):
    os.mkdir(log_path)
model_path = "./models/"
if not os.path.isdir(model_path):
    os.mkdir(model_path)

if len(sys.argv) == 6:
    with_speed = bool(int(sys.argv[1])) 
    with_profile = bool(int(sys.argv[2]))
    with_temporal = bool(int(sys.argv[3]))
    nn_sim_flag = bool(int(sys.argv[4]))
    input_type = sys.argv[5]
else:
    with_speed = True
    with_profile = True
    with_temporal = False
    nn_sim_flag = False
    input_type = ''
    
if input_type == 'all':
    input_type = ''
    
tag = '500_5_inputs_'+str(input_type)+'_speed'+str(with_speed)+'_profile'+\
str(with_profile)+'_temporal'+str(with_temporal)+'_nnsimilarity'+str(nn_sim_flag)
print(tag)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=log_path+tag+'.log',
                    filemode='a')

num_train_plates = 500
num_days = 5

# load data
path = '../data/data_profile/'

all_plates = load_data(path,'plates_2197.pkl')
train_plates_pool = all_plates[:2000]
train_plates = train_plates_pool[:num_train_plates]
test_plates = all_plates[2000:] # 197 plates

if with_profile:
    profile_data = load_data(path,'profile_features_11dim.pkl')
else:
    profile_data = []

if with_speed:
    raw_trajs = load_data(path,'mdp_trajs_allplates_201607_0415_1day2trajs_without_features_speed2.pkl')
else:
    raw_trajs = load_data(path,'mdp_trajs_allplates_201607_0415_1day2trajs_without_features.pkl')
    
    
if input_type == '' and with_speed == with_profile:
    # both speed and profile are True --> xyt+v+profile
    # both speed and profile are False --> xyt
    siamese_net = build_model_best(with_speed,with_profile,with_temporal,nn_sim_flag)
    get_pairs = get_pairs_s_and_d
elif input_type == '' and with_speed == True and with_profile == False:
    siamese_net = build_model_best(with_speed,with_profile,with_temporal,nn_sim_flag)
    get_pairs = get_pairs_s_and_d
elif input_type == '' and with_speed != with_profile:
    # profile is True and speed is False --> profile only (no xyt)
    siamese_net = build_model_profileonly()
    get_pairs = get_pairs_s_and_d
    raw_trajs = []
elif input_type != '':
    # when input_type != '' --> seek or drive 
    siamese_net = build_model_seekserve(with_speed,with_profile,with_temporal,nn_sim_flag)
    get_pairs = get_pairs_s_or_d
    
loss_500 = []
train_acc_500 = []
val_acc_500 = []
test_acc_500 = []
train_prob_list_500 = []
val_prob_list_500 = []
test_prob_list_500 = []

iteration = 1000000 
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
    
for ite in range(iteration): #epoch
    pair,label = get_pairs(raw_trajs, profile_data, train_plates, input_type, num_days)
    loss = siamese_net.train_on_batch(pair,label)
    pairs.append(pair)
    labels.append(label)
    if ite % 1000 == 0 and ite != 0:
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
        train_prob_list_500.append(train_prob_list)
        val_prob_list_500.append(val_prob_list)
        test_prob_list_500.append(test_prob_list)
        logging.info('******iteration: '+str(ite)+'; loss: '+str(loss)+ '; train acc: '+str(train_acc)+                     '; validation acc: '+ str(val_acc)+'; test acc: '+str(test_acc))
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
    if ite % 50000 == 0:
        save_model(siamese_net, model_path, tag = tag + '_mid')
        pickle.dump(loss_500, open(model_path+'loss_{0}.pkl'.format(tag),'wb'))
        pickle.dump(train_acc_500, open(model_path+'train_acc_{0}.pkl'.format(tag),'wb'))
        pickle.dump(val_acc_500, open(model_path+'val_acc_{0}.pkl'.format(tag),'wb'))
        pickle.dump(test_acc_500, open(model_path+'test_acc_{0}.pkl'.format(tag),'wb'))
        pickle.dump(train_prob_list_500, open(model_path+'train_probs_{0}.pkl'.format(tag),'wb'))
        pickle.dump(val_prob_list_500, open(model_path+'val_probs_{0}.pkl'.format(tag),'wb'))
        pickle.dump(test_prob_list_500, open(model_path+'test_probs_{0}.pkl'.format(tag),'wb'))
logging.info('total running time: '+ str(datetime.now()-t0))
save_model(siamese_net, model_path, tag = tag + '_iter'+str(iteration))
pickle.dump(loss_500, open(model_path+'loss_{0}.pkl'.format(tag),'wb'))
pickle.dump(train_acc_500, open(model_path+'train_acc_{0}.pkl'.format(tag),'wb'))
pickle.dump(val_acc_500, open(model_path+'val_acc_{0}.pkl'.format(tag),'wb'))
pickle.dump(test_acc_500, open(model_path+'test_acc_{0}.pkl'.format(tag),'wb'))
pickle.dump(train_prob_list_500, open(model_path+'train_probs_{0}.pkl'.format(tag),'wb'))
pickle.dump(val_prob_list_500, open(model_path+'val_probs_{0}.pkl'.format(tag),'wb'))
pickle.dump(test_prob_list_500, open(model_path+'test_probs_{0}.pkl'.format(tag),'wb'))
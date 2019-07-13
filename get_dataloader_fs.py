"""Data preprocessing for HAR"""

import h5py
import pickle
import numpy as np
import os
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
from matplotlib import cm 
from group_lasso import lls_l2l1

def draw_heatmap(data):
    #cmap=cm.Blues    
    cmap = cm.get_cmap('rainbow',1000)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])
    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    map = ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',
                  vmin=vmin,vmax=vmax)
    cb = plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
    plt.show()

def get_dataloader_fs(dataset, tgt_num_samp_per_class, src_type, k, tgtsbj):
    
    data_path = "./data/"+dataset+"/"+str(tgtsbj)+"/"

    # Get data labels for both source and target domains
    if src_type == "het":
        f_src = h5py.File(data_path+str(dataset)+'_srcsbj_srccls.h5', 'r')
        data_src = f_src.get('src_data')[()]
        labels_src = f_src.get('src_labels')[()]
    elif src_type == "hom":
        f_src = h5py.File(data_path+str(dataset)+'_tgtsbj_srccls.h5', 'r')
        data_src = f_src.get('src_data')[()]
        labels_src = f_src.get('src_labels')[()]
            
    f_tgt = h5py.File(data_path+str(dataset)+'_tgtsbj_tgtcls.h5', 'r')
    data_tgt = f_tgt.get('tgt_data')[()]
    labels_tgt = f_tgt.get('tgt_labels')[()]
    
    sample_id_path = "./sample_id/"+dataset+"/"+str(tgtsbj)+"/"
    if not os.path.exists(sample_id_path):
        os.makedirs(sample_id_path)
        
    # Adjust label values for both datasets to get successive integers starting from zero
    # Source
    labels_src_new = np.zeros(labels_src.shape)
    if dataset == "opp":
        for i in range(len(labels_src)):
            if labels_src[i] == 1:
                labels_src_new[i] = 0
            elif labels_src[i] == 3:
                labels_src_new[i] = 1
            elif labels_src[i] == 4:
                labels_src_new[i] = 2
            elif labels_src[i] == 7:
                labels_src_new[i] = 3
            elif labels_src[i] == 9:
                labels_src_new[i] = 4
            elif labels_src[i] == 11:
                labels_src_new[i] = 5
            elif labels_src[i] == 13:
                labels_src_new[i] = 6
            elif labels_src[i] == 14:
                labels_src_new[i] = 7
            elif labels_src[i] == 15:
                labels_src_new[i] = 8
            elif labels_src[i] == 16:
                labels_src_new[i] = 9
    if dataset == "pamap2":
        for i in range(len(labels_src)):
            if labels_src[i] == 0:
                labels_src_new[i] = 0
            elif labels_src[i] == 2:
                labels_src_new[i] = 1
            elif labels_src[i] == 3:
                labels_src_new[i] = 2
            elif labels_src[i] == 4:
                labels_src_new[i] = 3
            elif labels_src[i] == 7:
                labels_src_new[i] = 4
            elif labels_src[i] == 9:
                labels_src_new[i] = 5
            elif labels_src[i] == 11:
                labels_src_new[i] = 6
    labels_src = np.asarray(labels_src_new)
    labels_src_val = np.unique(labels_src)

    # Target
    labels_tgt_new = np.zeros(labels_tgt.shape)
    if dataset == "opp":
        for i in range(len(labels_tgt)):
            if labels_tgt[i] == 0:
                labels_tgt_new[i] = 0
            elif labels_tgt[i] == 2:
                labels_tgt_new[i] = 1
            elif labels_tgt[i] == 4:
                labels_tgt_new[i] = 2
            elif labels_tgt[i] == 6:
                labels_tgt_new[i] = 3
            elif labels_tgt[i] == 8:
                labels_tgt_new[i] = 4
            elif labels_tgt[i] == 10:
                labels_tgt_new[i] = 5
            elif labels_tgt[i] == 12:
                labels_tgt_new[i] = 6
    if dataset == "pamap2":
        for i in range(len(labels_tgt)):
            if labels_tgt[i] == 1:
                labels_tgt_new[i] = 0
            elif labels_tgt[i] == 5:
                labels_tgt_new[i] = 1
            elif labels_tgt[i] == 6:
                labels_tgt_new[i] = 2
            elif labels_tgt[i] == 8:
                labels_tgt_new[i] = 3
            elif labels_tgt[i] == 10:
                labels_tgt_new[i] = 4
    labels_tgt = np.asarray(labels_tgt_new)
    labels_tgt_val = np.unique(labels_tgt)
    tgt_num_samp_per_class_init = len(labels_tgt) // len(labels_tgt_val)
        
    # Training/testing set spliting for target domain
    data_train_tgt = []
    labels_train_tgt = []
    data_test_tgt = []
    labels_test_tgt = []
    train_size = int(tgt_num_samp_per_class)
    train_size_init = tgt_num_samp_per_class_init // 2
    for cls_id in labels_tgt_val:
        ind_for_clsi = np.array([i for i in range(len(labels_tgt)) if labels_tgt[i] == cls_id])
        indpool_train = [j for j in range(tgt_num_samp_per_class_init) if j%2==0]
        filename = sample_id_path+"indpool_train_"+str(k)+"_cls"+str(cls_id)+".txt"
        if not os.path.exists(filename):
            indpool_train = np.random.permutation(indpool_train).tolist()
            with open(filename, "wb") as fp:   
                pickle.dump(indpool_train, fp)
        else:
            with open(filename, "rb") as fp:   
                indpool_train = pickle.load(fp)
        indpool_test = [j for j in range(tgt_num_samp_per_class_init) if j%2==1]
        indpool = np.array(indpool_train + indpool_test)
        train = ind_for_clsi[indpool[:train_size]]
        test = ind_for_clsi[indpool[train_size_init:]]
        data_train_tgt.extend(data_tgt[train, :, :])
        labels_train_tgt.extend(labels_tgt[train])
        data_test_tgt.extend(data_tgt[test, :, :])
        labels_test_tgt.extend(labels_tgt[test])
    data_train_tgt = np.array(data_train_tgt)
    labels_train_tgt = np.array(labels_train_tgt)
    data_test_tgt = np.array(data_test_tgt)
    labels_test_tgt = np.array(labels_test_tgt)
    print("Target training data shape is: " + str(data_train_tgt.shape)) 
    print("Target testing data shape is: " + str(data_test_tgt.shape)) 
    
    # Data loader packing
    SHUFFLE = False
    # Target
    # Training set
    data_train_tgt_tensor = torch.FloatTensor(data_train_tgt)
    labels_train_tgt_tensor = torch.LongTensor(labels_train_tgt)
    tgt_torch_train = Data.TensorDataset(data_train_tgt_tensor, 
                                         labels_train_tgt_tensor)
    tgt_loader_train = Data.DataLoader(dataset=tgt_torch_train, 
                                       batch_size=len(tgt_torch_train), 
                                       shuffle=SHUFFLE)
    # Testing set
    data_test_tgt_tensor = torch.FloatTensor(data_test_tgt)
    labels_test_tgt_tensor = torch.LongTensor(labels_test_tgt)
    tgt_torch_test = Data.TensorDataset(data_test_tgt_tensor, 
                                        labels_test_tgt_tensor)
    tgt_loader_test = Data.DataLoader(dataset=tgt_torch_test, 
                                      batch_size=len(tgt_torch_test), 
                                      shuffle=SHUFFLE)
    # Source
    # Training set
    data_src_tensor = torch.FloatTensor(data_src)
    labels_src_tensor = torch.LongTensor(labels_src)
    src_torch = Data.TensorDataset(data_src_tensor, 
                                   labels_src_tensor)
    src_loader = Data.DataLoader(dataset=src_torch, 
                                 batch_size=len(src_torch), 
                                 shuffle=SHUFFLE)
    
                                      
    return src_loader, tgt_loader_train, tgt_loader_test
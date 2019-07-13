"""Pre-train encoder and classifier for source dataset."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import params
    
def train_single_domain_source(encoder, classifier, 
                               data_train,labels_train,
                               lr):
    
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer 
    params_to_train = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params_to_train,lr=lr)
    criterion_theta = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    loss_theta_curve = []
    
    # make data and labels variable
    # train
    data_train = Variable(data_train)
    labels_train = Variable(labels_train)
    
    for epoch in range(params.src_num_epochs_pre):
        
       # zero gradients for optimizer
        optimizer.zero_grad()

        # compute loss
        # loss_theta
        preds_train = classifier(encoder(data_train))
        loss_theta = criterion_theta(preds_train, labels_train)
        
        # optimize source classifier
        loss_theta.backward()
        optimizer.step()
        
        # Collect loss values
        loss_theta_curve.extend([loss_theta.item()])

    return encoder, classifier
    
def train_single_domain_target(encoder, classifier, 
                               data_train,labels_train,
                               data_test,labels_test,
                               lr):
    
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer 
    params_to_train = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params_to_train,lr=lr)
    criterion_theta = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    loss_theta_curve = []
    f1score_curve = []
    num_samp_test = len(labels_test)
    pred_mat = np.zeros((num_samp_test,params.tgt_num_epochs_pre))
    
    # make data and labels variable
    # train
    data_train = Variable(data_train)
    labels_train = Variable(labels_train)
    # test
    data_test = Variable(data_test)
    labels_test = Variable(labels_test)    
    
    for epoch in range(params.tgt_num_epochs_pre):
        
       # zero gradients for optimizer
        optimizer.zero_grad()

        # compute loss
        # loss_theta
        preds_train = classifier(encoder(data_train))
        loss_theta = criterion_theta(preds_train, labels_train)
        
        # optimize source classifier
        loss_theta.backward()
        optimizer.step()
        
        # Collect loss values
        loss_theta_curve.extend([loss_theta.item()])

        # eval model on test set  
        f1score, pred = eval_single_domain(encoder, classifier, data_test, labels_test)        
        f1score_curve.extend([f1score])
        pred_mat[:,epoch] = pred
    
    f1score_final = max(f1score_curve)
    f1score_final_ind = np.argmax(f1score_curve)
    print('f1score='+str(int(f1score_final))+', iter='+str(np.argmax(f1score_curve)))
    pred_final = pred_mat[:,f1score_final_ind]
    confusemat_final = confusion_matrix(labels_test,pred_final)

    return encoder, classifier, f1score_final, confusemat_final
    

def eval_single_domain(encoder, classifier, data, labels):
    
    encoder.eval()    
    classifier.eval()

    # evaluate network
    data = Variable(data)
    labels = Variable(labels)

    outputs = classifier(encoder(data))
    
    _, pred = torch.max(outputs.data, 1)
    
    labels = np.squeeze(labels.numpy())
    num_class = len(np.unique(labels))
    acc_mat = np.zeros(num_class)
    for cls_id in range(num_class):
        ind = np.where(labels==cls_id)
        labels_i = labels[ind]
        pred_i = pred[ind]
        acc_mat[cls_id] = \
        (np.expand_dims(pred_i,axis=1) == np.expand_dims(labels_i,axis=1)).sum()/len(labels_i)

    f1score = f1_score(labels,pred,average='macro') * 100
    
    return f1score, pred
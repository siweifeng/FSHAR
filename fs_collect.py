import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
import h5py
from opp2opp_fs import main_opp2opp_fs
from pa2pa_fs import main_pa2pa_fs

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


"""opp"""

tgt_num_samp_per_class_list = [1,5]
K = 2
num_exp = 4
tgtsbj_list = [1,2,3,4]
src_type_list = ["hom","het"]
sim_type_list = ["SR","NGD","Cos"]

TGT_NUM_CLASSES = 7

f1score_list_general = np.zeros((num_exp,K))
f1score_list_general_avg = np.zeros((num_exp))
f1score_list_general_std = np.zeros((num_exp))

confusemat_list_general = np.zeros((TGT_NUM_CLASSES,TGT_NUM_CLASSES,num_exp,K))
confusemat_list_general_avg = np.zeros((TGT_NUM_CLASSES,TGT_NUM_CLASSES,num_exp))

count = -1

for sim_type in sim_type_list:
    for src_type in src_type_list:
        for tgtsbj in tgtsbj_list:
            for tgt_num_samp_per_class in tgt_num_samp_per_class_list:
                rslt_path = "./results(f1)_"+sim_type+"/opp/"+src_type+\
                            "/tgtsbj="+str(tgtsbj)+"/num_samp="+\
                            str(tgt_num_samp_per_class)+"/"
                if not os.path.exists(rslt_path):
                    os.makedirs(rslt_path)
                if not os.path.exists(rslt_path+"f1score_stat.h5"):
                    for k in range(K):
                        k = int(k)
                        print('k = '+str(k))
                        f1score_list, confusemat_list = \
                        main_opp2opp_fs(tgt_num_samp_per_class,src_type,sim_type,k,tgtsbj)
                        f1score_list_general[:,k] = f1score_list.tolist()
                        confusemat_list_general[:,:,:,k] = confusemat_list
                        print(f1score_list_general)
                    for i in range(num_exp):
                        f1score_list_general_avg[i] = \
                        np.mean(np.squeeze(f1score_list_general[i,:])).tolist()
                        f1score_list_general_std[i] = \
                        np.std(np.squeeze(f1score_list_general[i,:])).tolist()
                        confusemat_list_general_avg[:,:,i] = \
                        np.mean(confusemat_list_general[:,:,i,:],axis=2)
                        print("The result for the "+str(i)+"th f1 score is: "+\
                        str(int(round(f1score_list_general_avg[i])))+"+\-"+\
                        str(int(round(f1score_list_general_std[i]))))
                    # Save statistics as plot
                    count = count + 1
                    plt.figure(count)
                    plt.errorbar(list(range(0,num_exp)),f1score_list_general_avg,
                                 f1score_list_general_std,marker='s',mfc=None,
                                 mec='blue',ms=5,mew=2)
                    plt.xlim((-0.1,num_exp+0.1))
                    plt.savefig(rslt_path+"f1score_stat.png")
                    plt.clf()
                    plt.close()
                    # Save statistics as file
                    f = h5py.File(rslt_path+"f1score_stat.h5")
                    f.create_dataset("avg", data=f1score_list_general_avg)
                    f.create_dataset("std", data=f1score_list_general_std)
                    f.close()
                    print('Done.')
                    # Save all results as file
                    f = h5py.File(rslt_path+"f1score.h5")
                    f.create_dataset("list", data=f1score_list_general)
                    f.close()
                    print('Done.')
                    # Save statistics as file
                    f = h5py.File(rslt_path+"confusemat_stat.h5")
                    f.create_dataset("avg", data=confusemat_list_general_avg)
                    f.close()
                    print('Done.')
  

"""pamap2"""
                      
tgt_num_samp_per_class_list = [1,5]
K = 2
num_exp = 4
tgtsbj_list = [1,2,3]
src_type_list = ["hom","het"]
sim_type_list = ["SR","NGD","Cos"]

TGT_NUM_CLASSES = 5
    
acc_list_general = np.zeros((num_exp,K))
acc_list_general_avg = np.zeros((num_exp))
acc_list_general_std = np.zeros((num_exp))

f1score_list_general = np.zeros((num_exp,K))
f1score_list_general_avg = np.zeros((num_exp))
f1score_list_general_std = np.zeros((num_exp))

confusemat_list_general = np.zeros((TGT_NUM_CLASSES,TGT_NUM_CLASSES,num_exp,K))
confusemat_list_general_avg = np.zeros((TGT_NUM_CLASSES,TGT_NUM_CLASSES,num_exp))

count = -1

for sim_type in sim_type_list:
    for src_type in src_type_list:
        for tgtsbj in tgtsbj_list:
            for tgt_num_samp_per_class in tgt_num_samp_per_class_list:
                rslt_path = "./results(f1)_"+sim_type+"/pamap2/"+src_type+\
                            "/tgtsbj="+str(tgtsbj)+"/num_samp="+\
                            str(tgt_num_samp_per_class)+"/"
                if not os.path.exists(rslt_path):
                    os.makedirs(rslt_path)
                if not os.path.exists(rslt_path+"acc_stat.h5"):
                    for k in range(K):
                        k = int(k)
                        print('k = '+str(k))
                        f1score_list, confusemat_list = \
                        main_pa2pa_fs(tgt_num_samp_per_class,src_type,sim_type,k,tgtsbj)
                        f1score_list_general[:,k] = f1score_list.tolist()
                        confusemat_list_general[:,:,:,k] = confusemat_list
                        print(f1score_list_general)
                    for i in range(num_exp):
                        f1score_list_general_avg[i] = \
                        np.mean(np.squeeze(f1score_list_general[i,:])).tolist()
                        f1score_list_general_std[i] = \
                        np.std(np.squeeze(f1score_list_general[i,:])).tolist()
                        confusemat_list_general_avg[:,:,i] = \
                        np.mean(confusemat_list_general[:,:,i,:],axis=2)
                        print("The result for the "+str(i)+"th f1 score is: "+\
                        str(int(round(f1score_list_general_avg[i])))+"+\-"+\
                        str(int(round(f1score_list_general_std[i]))))
                    # Save statistics as plot
                    count = count + 1
                    plt.figure(count)
                    plt.errorbar(list(range(0,num_exp)),f1score_list_general_avg,
                                 f1score_list_general_std,marker='s',mfc=None,
                                 mec='blue',ms=5,mew=2)
                    plt.xlim((-0.1,num_exp+0.1))
                    plt.savefig(rslt_path+"f1score_stat.png")
                    plt.clf()
                    plt.close()
                    # Save statistics as file
                    f = h5py.File(rslt_path+"f1score_stat.h5")
                    f.create_dataset("avg", data=f1score_list_general_avg)
                    f.create_dataset("std", data=f1score_list_general_std)
                    f.close()
                    print('Done.')
                    # Save all results as file
                    f = h5py.File(rslt_path+"f1score.h5")
                    f.create_dataset("list", data=f1score_list_general)
                    f.close()
                    print('Done.')
                    # Save statistics as file
                    f = h5py.File(rslt_path+"confusemat_stat.h5")
                    f.create_dataset("avg", data=confusemat_list_general_avg)
                    f.close()
                    print('Done.')
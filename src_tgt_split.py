import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt

dataset = sys.argv[1]
tgtsbj = sys.argv[2]
data_path = "./data/"+str(dataset)+"/"+tgtsbj+"/"

if dataset == "opp":
    label_map = [
                (0, 'Open Door 1'),
                (1, 'Open Door 2'),
                (2, 'Close Door 1'),
                (3, 'Close Door 2'),
                (4, 'Open Fridge'),
                (5, 'Close Fridge'),
                (6, 'Open Dishwasher'),
                (7, 'Close Dishwasher'),
                (8, 'Open Drawer 1'),
                (9, 'Close Drawer 1'),
                (10, 'Open Drawer 2'),
                (11, 'Close Drawer 2'),
                (12, 'Open Drawer 3'),
                (13, 'Close Drawer 3'),
                (14, 'Clean Table'),
                (15, 'Drink from Cup'),
                (16, 'Toggle Switch')
                ]               
    tgt_cls = [0,2,4,6,8,10,12]
    src_cls = [1,3,5,7,9,11,13,14,15,16]
elif dataset == "pamap2":
    label_map = [
                (0, 'lying'),
                (1, 'sitting'),
                (2, 'standing'),
                (3, 'walking'),
                (4, 'running'),
                (5, 'cycling'),
                (6, 'Nordic walking'),
                (7, 'ascending stairs'),
                (8, 'descending stairs'),
                (9, 'vacuum cleaning'),
                (10, 'ironing'),
                (11, 'rope jumping')
                ]
    tgt_cls = [1,5,6,8,10]
    src_cls = [0,2,3,4,7,9,11]
    
# Source Subject
    
f_srcsbj = h5py.File(data_path+dataset+"_srcsbj_seg_balcls.h5", 'r')
data_srcsbj_seg_balcls = f_srcsbj.get('data_srcsbj_seg_balcls')[()]
labels_srcsbj_seg_balcls = f_srcsbj.get('labels_srcsbj_seg_balcls')[()]
print("Dataset is of size " + str(data_srcsbj_seg_balcls.shape))
print("Label set is of size" + str(labels_srcsbj_seg_balcls.shape))

src_idx = []
for cls_id in src_cls:
    ind_for_clsi = [i for i in range(len(labels_srcsbj_seg_balcls)) if labels_srcsbj_seg_balcls[i] == cls_id]
    src_idx.extend(ind_for_clsi)
src_data = data_srcsbj_seg_balcls[src_idx, :, :]
src_labels = labels_srcsbj_seg_balcls[src_idx]
#plt.plot(src_labels)
#plt.show()
print("Source data labels are: " + str(np.unique(src_labels)))
print("Source data size is: " + str(src_data.shape))

tgt_idx = []
for cls_id in tgt_cls:
    ind_for_clsi = [i for i in range(len(labels_srcsbj_seg_balcls)) if labels_srcsbj_seg_balcls[i] == cls_id]
    tgt_idx.extend(ind_for_clsi)    
tgt_data = data_srcsbj_seg_balcls[tgt_idx, :, :]
tgt_labels = labels_srcsbj_seg_balcls[tgt_idx]  
#plt.plot(tgt_labels)
#plt.show()
print("Target data labels are: " + str(np.unique(tgt_labels))) 
print("Target data size is: " + str(tgt_data.shape))

f_srcsbj = h5py.File(data_path+dataset+'_srcsbj_srccls.h5')
f_srcsbj.create_dataset('src_data', data=src_data)
f_srcsbj.create_dataset('src_labels', data=src_labels)
f_srcsbj.close()
print('Done for source.')
f_srcsbj = h5py.File(data_path+dataset+'_srcsbj_tgtcls.h5')
f_srcsbj.create_dataset('tgt_data', data=tgt_data)
f_srcsbj.create_dataset('tgt_labels', data=tgt_labels)
f_srcsbj.close()
print('Done for target.')

# Target Subject

f_tgtsbj = h5py.File(data_path+dataset+"_tgtsbj_seg_balcls.h5", 'r')
data_tgtsbj_seg_balcls = f_tgtsbj.get('data_tgtsbj_seg_balcls')[()]
labels_tgtsbj_seg_balcls = f_tgtsbj.get('labels_tgtsbj_seg_balcls')[()]
print("Dataset is of size " + str(data_tgtsbj_seg_balcls.shape))
print("Label set is of size" + str(labels_tgtsbj_seg_balcls.shape))

src_idx = []
for cls_id in src_cls:
    ind_for_clsi = [i for i in range(len(labels_tgtsbj_seg_balcls)) if labels_tgtsbj_seg_balcls[i] == cls_id]
    src_idx.extend(ind_for_clsi)
src_data = data_tgtsbj_seg_balcls[src_idx, :, :]
src_labels = labels_tgtsbj_seg_balcls[src_idx]
#plt.plot(src_labels)
#plt.show()
print("Source data labels are: " + str(np.unique(src_labels)))
print("Source data size is: " + str(src_data.shape))

tgt_idx = []
for cls_id in tgt_cls:
    ind_for_clsi = [i for i in range(len(labels_tgtsbj_seg_balcls)) if labels_tgtsbj_seg_balcls[i] == cls_id]
    tgt_idx.extend(ind_for_clsi)    
tgt_data = data_tgtsbj_seg_balcls[tgt_idx, :, :]
tgt_labels = labels_tgtsbj_seg_balcls[tgt_idx] 
#plt.plot(tgt_labels)
#plt.show() 
print("Target data labels are: " + str(np.unique(tgt_labels))) 
print("Target data size is: " + str(tgt_data.shape))

f_tgtsbj = h5py.File(data_path+dataset+'_tgtsbj_srccls.h5')
f_tgtsbj.create_dataset('src_data', data=src_data)
f_tgtsbj.create_dataset('src_labels', data=src_labels)
f_tgtsbj.close()
print('Done for source.')
f_tgtsbj = h5py.File(data_path+dataset+'_tgtsbj_tgtcls.h5')
f_tgtsbj.create_dataset('tgt_data', data=tgt_data)
f_tgtsbj.create_dataset('tgt_labels', data=tgt_labels)
f_tgtsbj.close()
print('Done for target.')
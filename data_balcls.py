import matplotlib.pyplot as plt
from matplotlib import cm 
import h5py
import sys
import numpy as np

#def draw_heatmap(data):
#    #cmap=cm.Blues    
#    cmap=cm.get_cmap('rainbow',1000)
#    figure=plt.figure(facecolor='w')
#    ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])
#    vmax=data[0][0]
#    vmin=data[0][0]
#    for i in data:
#        for j in i:
#            if j>vmax:
#                vmax=j
#            if j<vmin:
#                vmin=j
#    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',
#                  vmin=vmin,vmax=vmax)
#    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
#    plt.show()

dataset = sys.argv[1]
tgtsbj = sys.argv[2]
    
data_path = "./data/"+str(dataset)+"/"+tgtsbj+"/"

# Source Subject
    
f_srcsbj = h5py.File(data_path+str(dataset)+"_srcsbj_seg.h5", 'r')
data_srcsbj_seg = f_srcsbj.get('data_srcsbj_seg')[()]
labels_srcsbj_seg = f_srcsbj.get('labels_srcsbj_seg')[()]
print("Dataset is of size " + str(data_srcsbj_seg.shape))
print("Label set is of size" + str(labels_srcsbj_seg.shape))
label_srcsbj_seg_freq = np.bincount(labels_srcsbj_seg)[1:]
print("Label frequency is " + str(label_srcsbj_seg_freq))

num_cls = max(labels_srcsbj_seg)+1
print("There are "+str(num_cls)+" classes for "+dataset)
a = np.zeros((num_cls))
for i in range(len(labels_srcsbj_seg)):
    a[labels_srcsbj_seg[i]] = a[labels_srcsbj_seg[i]] + 1
for j in range(len(a)):
    if a[j] == 0:
        a[j] = 1e5
numsamp_per_cls = int(min(a))
print("Sample number for each class is adjusted to "+str(numsamp_per_cls))
    
data_srcsbj_seg_balcls = []
labels_srcsbj_seg_balcls = []
labels_srcsbj_seg_uniq = np.unique(labels_srcsbj_seg)
print("Labels are " + str(labels_srcsbj_seg_uniq))
for cls_id in np.unique(labels_srcsbj_seg):
    if cls_id != 0:
        ind_for_clsi = np.array([ind for ind in range(len(labels_srcsbj_seg)) if labels_srcsbj_seg[ind] == cls_id])
        temp = np.random.permutation(len(ind_for_clsi))
        ind_for_clsi_bal = ind_for_clsi[temp[:numsamp_per_cls]]
        data_srcsbj_seg_balcls.extend(data_srcsbj_seg[ind_for_clsi_bal,:,:])
        labels_srcsbj_seg_balcls.extend(labels_srcsbj_seg[ind_for_clsi_bal])

data_srcsbj_seg_balcls = np.array(data_srcsbj_seg_balcls)
if dataset == "opp":
    labels_srcsbj_seg_balcls = [labels_srcsbj_seg_balcls[i] - 1 for i in range(len(labels_srcsbj_seg_balcls))]
    labels_srcsbj_seg_balcls = np.array(labels_srcsbj_seg_balcls)
elif dataset == "pamap2":
    for i in range(len(labels_srcsbj_seg_balcls)):
        if labels_srcsbj_seg_balcls[i] >=1 and labels_srcsbj_seg_balcls[i] <= 7:
            labels_srcsbj_seg_balcls[i] = labels_srcsbj_seg_balcls[i] - 1
        elif labels_srcsbj_seg_balcls[i] >= 11 and labels_srcsbj_seg_balcls[i] <=14:
            labels_srcsbj_seg_balcls[i] = labels_srcsbj_seg_balcls[i] - 4
        elif labels_srcsbj_seg_balcls[i] == 18:
            labels_srcsbj_seg_balcls[i] = 11
    labels_srcsbj_seg_balcls = np.array(labels_srcsbj_seg_balcls)

print("Class-balanced dataset is of size " + str(data_srcsbj_seg_balcls.shape))
print("Class-balanced label set is of size" + str(labels_srcsbj_seg_balcls.shape))

a = np.zeros((len(np.unique(labels_srcsbj_seg_balcls))))
for i in range(len(labels_srcsbj_seg_balcls)):
    a[labels_srcsbj_seg_balcls[i]] = a[labels_srcsbj_seg_balcls[i]] + 1
for j in range(len(a)):
    print("Class #"+str(j)+" has "+str(a[j])+" samples.")

data_srcsbj_seg_balcls_forvisu = np.mean(data_srcsbj_seg_balcls, 1)
#draw_heatmap(data_srcsbj_seg_balcls_forvisu)
#plt.plot(labels_srcsbj_seg_balcls)
#plt.show()

f_srcsbj = h5py.File(data_path+str(dataset)+"_srcsbj_seg_balcls.h5")
f_srcsbj.create_dataset('data_srcsbj_seg_balcls', data=data_srcsbj_seg_balcls)
f_srcsbj.create_dataset('labels_srcsbj_seg_balcls', data=labels_srcsbj_seg_balcls)
f_srcsbj.close()
print('Done.')

# Target Subject

f_tgtsbj = h5py.File(data_path+str(dataset)+"_tgtsbj_seg.h5", 'r')
data_tgtsbj_seg = f_tgtsbj.get('data_tgtsbj_seg')[()]
labels_tgtsbj_seg = f_tgtsbj.get('labels_tgtsbj_seg')[()]
print("Dataset is of size " + str(data_tgtsbj_seg.shape))
print("Label set is of size" + str(labels_tgtsbj_seg.shape))
label_tgtsbj_seg_freq = np.bincount(labels_tgtsbj_seg)[1:]
print("Label frequency is " + str(label_tgtsbj_seg_freq))

num_cls = max(labels_tgtsbj_seg)+1
print("There are "+str(num_cls)+" classes for "+dataset)
a = np.zeros((num_cls))
for i in range(len(labels_tgtsbj_seg)):
    a[labels_tgtsbj_seg[i]] = a[labels_tgtsbj_seg[i]] + 1
for j in range(len(a)):
    if a[j] == 0:
        a[j] = 1e5
numsamp_per_cls = int(min(a))
print("Sample number for each class is adjusted to "+str(numsamp_per_cls))
    
data_tgtsbj_seg_balcls = []
labels_tgtsbj_seg_balcls = []
labels_tgtsbj_seg_uniq = np.unique(labels_tgtsbj_seg)
print("Labels are " + str(labels_tgtsbj_seg_uniq))
for cls_id in np.unique(labels_tgtsbj_seg):
    if cls_id != 0:
        ind_for_clsi = np.array([ind for ind in range(len(labels_tgtsbj_seg)) if labels_tgtsbj_seg[ind] == cls_id])
        temp = np.random.permutation(len(ind_for_clsi))
        ind_for_clsi_bal = ind_for_clsi[temp[:numsamp_per_cls]]
        data_tgtsbj_seg_balcls.extend(data_tgtsbj_seg[ind_for_clsi_bal,:,:])
        labels_tgtsbj_seg_balcls.extend(labels_tgtsbj_seg[ind_for_clsi_bal])

data_tgtsbj_seg_balcls = np.array(data_tgtsbj_seg_balcls)
if dataset == "opp":
    labels_tgtsbj_seg_balcls = [labels_tgtsbj_seg_balcls[i] - 1 for i in range(len(labels_tgtsbj_seg_balcls))]
    labels_tgtsbj_seg_balcls = np.array(labels_tgtsbj_seg_balcls)
elif dataset == "pamap2":
    for i in range(len(labels_tgtsbj_seg_balcls)):
        if labels_tgtsbj_seg_balcls[i] >=1 and labels_tgtsbj_seg_balcls[i] <= 7:
            labels_tgtsbj_seg_balcls[i] = labels_tgtsbj_seg_balcls[i] - 1
        elif labels_tgtsbj_seg_balcls[i] >= 11 and labels_tgtsbj_seg_balcls[i] <=14:
            labels_tgtsbj_seg_balcls[i] = labels_tgtsbj_seg_balcls[i] - 4
        elif labels_tgtsbj_seg_balcls[i] == 18:
            labels_tgtsbj_seg_balcls[i] = 11
    labels_tgtsbj_seg_balcls = np.array(labels_tgtsbj_seg_balcls)

print("Class-balanced dataset is of size " + str(data_tgtsbj_seg_balcls.shape))
print("Class-balanced label set is of size" + str(labels_tgtsbj_seg_balcls.shape))

a = np.zeros((len(np.unique(labels_tgtsbj_seg_balcls))))
for i in range(len(labels_tgtsbj_seg_balcls)):
    a[labels_tgtsbj_seg_balcls[i]] = a[labels_tgtsbj_seg_balcls[i]] + 1
for j in range(len(a)):
    print("Class #"+str(j)+" has "+str(a[j])+" samples.")

data_tgtsbj_seg_balcls_forvisu = np.mean(data_tgtsbj_seg_balcls, 1)
#draw_heatmap(data_tgtsbj_seg_balcls_forvisu)
#plt.plot(labels_tgtsbj_seg_balcls)
#plt.show()

f_tgtsbj = h5py.File(data_path+str(dataset)+"_tgtsbj_seg_balcls.h5")
f_tgtsbj.create_dataset('data_tgtsbj_seg_balcls', data=data_tgtsbj_seg_balcls)
f_tgtsbj.create_dataset('labels_tgtsbj_seg_balcls', data=labels_tgtsbj_seg_balcls)
f_tgtsbj.close()
print('Done.')
import numpy as np
from scipy import stats
import h5py
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += (size / 2)

def segment_opp(x_train, y_train, window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1, window_size, 77))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        start = round(start)
        end = round(end)
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels
    
def segment_pamap2(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,52))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        start = round(start)
        end = round(end)
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels
    
dataset = sys.argv[1]
tgtsbj = sys.argv[2]
data_path = "./data/"+str(dataset)+"/"+tgtsbj+"/"

# Source Subject

f_srcsbj = h5py.File(data_path+dataset+"_srcsbj.h5", 'r')

data_srcsbj = f_srcsbj.get('data')[()]
labels_srcsbj = f_srcsbj.get('labels')[()]
labels_srcsbj = np.asarray([x - 1 for x in labels_srcsbj])
labels_srcsbj = labels_srcsbj.astype(np.int64)

print("data size = " + str(data_srcsbj.shape))
print("labels size =" + str(labels_srcsbj.shape))
a = np.zeros((max(labels_srcsbj)+1))
for i in range(len(labels_srcsbj)):
    a[labels_srcsbj[i]] = a[labels_srcsbj[i]] + 1
for j in range(len(a)):
    print("Class #"+str(j)+" has "+str(a[j])+" samples.")

if dataset == "opp":
    input_width = 23
    print("segmenting signal...")
    data_srcsbj_seg, labels_srcsbj_seg\
    = segment_opp(data_srcsbj, labels_srcsbj, input_width)
    labels_srcsbj_seg = labels_srcsbj_seg.astype(np.int64)
    print("signal segmented.")
elif dataset =="pamap2":
    input_width = 25
    print("segmenting signal...")
    data_srcsbj_seg, labels_srcsbj_seg\
    = segment_pamap2(data_srcsbj, labels_srcsbj, input_width)
    labels_srcsbj_seg = labels_srcsbj_seg.astype(np.int64)
    print("signal segmented.")
else:
    print("no correct dataset")
    exit(0)
    
print("data_srcsbj_seg size =" + str(data_srcsbj_seg.shape))
print("labels_srcsbj_seg size =" + str(labels_srcsbj_seg.shape))
a = np.zeros((max(labels_srcsbj)+1))
for i in range(len(labels_srcsbj_seg)):
    a[labels_srcsbj_seg[i]] = a[labels_srcsbj_seg[i]] + 1
for j in range(len(a)):
    print("Class #"+str(j)+" has "+str(a[j])+" samples.")
    
f_srcsbj = h5py.File(data_path+dataset+"_srcsbj_seg.h5")
f_srcsbj.create_dataset('data_srcsbj_seg', data=data_srcsbj_seg)
f_srcsbj.create_dataset('labels_srcsbj_seg', data=labels_srcsbj_seg)
f_srcsbj.close()
print('Done.')
    
# Target Subject
    
f_tgtsbj = h5py.File(data_path+dataset+"_tgtsbj.h5", 'r')

data_tgtsbj = f_tgtsbj.get('data')[()]
labels_tgtsbj = f_tgtsbj.get('labels')[()]
labels_tgtsbj = np.asarray([x - 1 for x in labels_tgtsbj])
labels_tgtsbj = labels_tgtsbj.astype(np.int64)

print("data size = " + str(data_tgtsbj.shape))
print("labels size =" + str(labels_tgtsbj.shape))
a = np.zeros((max(labels_tgtsbj)+1))
for i in range(len(labels_tgtsbj)):
    a[labels_tgtsbj[i]] = a[labels_tgtsbj[i]] + 1
for j in range(len(a)):
    print("Class #"+str(j)+" has "+str(a[j])+" samples.")

if dataset == "opp":
    input_width = 23
    print("segmenting signal...")
    data_tgtsbj_seg, labels_tgtsbj_seg\
    = segment_opp(data_tgtsbj, labels_tgtsbj, input_width)
    labels_tgtsbj_seg = labels_tgtsbj_seg.astype(np.int64)
    print("signal segmented.")
elif dataset =="pamap2":
    input_width = 25
    print("segmenting signal...")
    data_tgtsbj_seg, labels_tgtsbj_seg\
    = segment_pamap2(data_tgtsbj, labels_tgtsbj, input_width)
    labels_tgtsbj_seg = labels_tgtsbj_seg.astype(np.int64)
    print("signal segmented.")
else:
    print("no correct dataset")
    exit(0)
    
print("data_tgtsbj_seg size =" + str(data_tgtsbj_seg.shape))
print("labels_tgtsbj_seg size =" + str(labels_tgtsbj_seg.shape))
a = np.zeros((max(labels_tgtsbj)+1))
for i in range(len(labels_tgtsbj_seg)):
    a[labels_tgtsbj_seg[i]] = a[labels_tgtsbj_seg[i]] + 1
for j in range(len(a)):
    print("Class #"+str(j)+" has "+str(a[j])+" samples.")
    
f_tgtsbj = h5py.File(data_path+dataset+"_tgtsbj_seg.h5")
f_tgtsbj.create_dataset('data_tgtsbj_seg', data=data_tgtsbj_seg)
f_tgtsbj.create_dataset('labels_tgtsbj_seg', data=labels_tgtsbj_seg)
f_tgtsbj.close()
print('Done.')
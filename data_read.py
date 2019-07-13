import numpy as np
import csv
import sys
import h5py
import os

class data_reader:
    
    def __init__(self, dataset,tgtsbj):
        if dataset =="opp":
            self.data_srcsbj, self.data_tgtsbj, self.idToLabel = self.readOpportunity(tgtsbj)
        elif dataset == "pamap2":
            self.data_srcsbj, self.data_tgtsbj, self.idToLabel = self.readPamap2(tgtsbj)
        else:
            print('Not supported yet')
            sys.exit(0)
        self.save_data(dataset,tgtsbj)
            
    def save_data(self,dataset,tgtsbj):
        data_path = "./data/"+str(dataset)+"/"+tgtsbj+"/"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        f_srcsbj = h5py.File(data_path+str(dataset)+"_srcsbj.h5")
        for sbj in self.data_srcsbj:
            f_srcsbj.create_dataset(sbj, data=self.data_srcsbj[sbj])
        f_srcsbj.close()
        f_tgtsbj = h5py.File(data_path+str(dataset)+"_tgtsbj.h5")
        for sbj in self.data_tgtsbj:
            f_tgtsbj.create_dataset(sbj, data=self.data_tgtsbj[sbj])
        f_tgtsbj.close()
        print('Done.')
            
    def readOpportunity(self,tgtsbj):
        tgtsbj = int(tgtsbj)
        if tgtsbj == 1:
            files_srcsbj = ['S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 
            'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat', 
            'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 
            'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat', 
            'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 
            'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']
            files_tgtsbj = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 
            'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat']
        elif tgtsbj == 2:
            files_srcsbj = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 
            'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 
            'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 
            'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat', 
            'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 
            'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']
            files_tgtsbj = ['S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 
            'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat']
        elif tgtsbj == 3:
            files_srcsbj = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 
            'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 
            'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 
            'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat', 
            'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 
            'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']
            files_tgtsbj = ['S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 
            'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat']
        elif tgtsbj == 4:
            files_srcsbj = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 
            'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 
            'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 
            'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat', 
            'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 
            'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat']
            files_tgtsbj = ['S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 
            'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']
        #names are from label_legend.txt of Opportunity dataset
        #except 0-ie Other, which is an additional label
        label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        cols = [
            37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 56, 
            57, 58,63, 64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 80, 81, 
            82, 83, 84, 89, 90, 91, 92, 93, 94, 95, 96, 97, 102, 103, 104, 
            105, 106, 107, 108,109, 110, 111, 112, 113, 114, 115, 116, 117, 
            118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 
            131, 132, 133, 249
            ]

        data_srcsbj = self.readOpportunityFiles(files_srcsbj, cols, labelToId)
        data_tgtsbj = self.readOpportunityFiles(files_tgtsbj, cols, labelToId)

        return data_srcsbj, data_tgtsbj, idToLabel
        
    def readOpportunityFiles(self, filelist, cols, labelToId):
        data = []
        labels = []
        i = 0
        for filename in filelist:
            print('Reading file %d of %d with name %s' % 
            (i+1, len(filelist), filename))
            i = i + 1
            with open('./opp/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                count_line = 0
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        count_line = count_line + 1
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])   
                print("There are totally " + str(count_line) + " lines in " + 
                filename + ".")
        return {'data': np.asarray(data),
                'labels': np.asarray(labels, dtype=int) + 1}
                
                
    def readPamap2(self,tgtsbj):
        tgtsbj = int(tgtsbj)
        if tgtsbj == 1:
            files_srcsbj = ['subject102.dat', 'subject103.dat', 'subject105.dat',
                            'subject106.dat', 'subject108.dat', 'subject109.dat']
            files_tgtsbj = ['subject101.dat', 'subject104.dat','subject107.dat']
        elif tgtsbj == 2:
            files_srcsbj = ['subject101.dat', 'subject103.dat', 'subject104.dat',
                            'subject106.dat', 'subject107.dat', 'subject109.dat']
            files_tgtsbj = ['subject102.dat', 'subject105.dat', 'subject108.dat']
        elif tgtsbj == 3:
            files_srcsbj = ['subject101.dat', 'subject102.dat', 'subject104.dat',
                            'subject105.dat', 'subject107.dat', 'subject108.dat']
            files_tgtsbj = ['subject103.dat', 'subject106.dat', 'subject109.dat']
        label_map = [
            (0, 'other'),#0
            (1, 'lying'),#1
            (2, 'sitting'),#2
            (3, 'standing'),#3
            (4, 'walking'),#4
            (5, 'running'),#5
            (6, 'cycling'),#6
            (7, 'Nordic walking'),#7
            (9, 'watching TV'),#8
            (10, 'computer work'),#9
            (11, 'car driving'),#10
            (12, 'ascending stairs'),#11
            (13, 'descending stairs'),#12
            (16, 'vacuum cleaning'),#13
            (17, 'ironing'),#14
            (18, 'folding laundry'),#15
            (19, 'house cleaning'),#16
            (20, 'playing soccer'),#17
            (24, 'rope jumping')#18
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        
        idToLabel = [x[1] for x in label_map]
        
        cols = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 
                49, 50, 51, 52, 53
               ]
        
        data_srcsbj = self.readPamap2Files(files_srcsbj, cols, labelToId)
        data_tgtsbj = self.readPamap2Files(files_tgtsbj, cols, labelToId)
                    
        return data_srcsbj, data_tgtsbj, idToLabel

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            with open('./pamap2/Protocol/%s' % filename, 'r') as f:
                #print "f",f
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    #print "line=",line
                    elem = []
                    #not including the non related activity
                    if line[1] == "0":
                        continue
                    # if line[10] == "0":
                    #     continue
                    for ind in cols:
                        #print "ind=",ind
                        # if ind == 10:
                        #     # print "line[ind]",line[ind]
                        #     if line[ind] == "0":
                        #         continue
                        elem.append(line[ind])
                    # print "elem =",elem
                    # print "elem[:-1] =",elem[:-1]
                    # print "elem[0] =",elem[0]
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[0]])
                        # print "[x for x in elem[:-1]]=",[x for x in elem[:-1]]
                        # print "[float(x) / 1000 for x in elem[:-1]]=",[float(x) / 1000 for x in elem[:-1]]
                        # print "labelToId[elem[0]]=",labelToId[elem[0]]
                        # print "labelToId[elem[-1]]",labelToId[elem[-1]]
# sys.exit(0)
                        
        return {'data': np.asarray(data), 
                'labels': np.asarray(labels, dtype=int)+1}

if __name__ == "__main__":
    print('Reading %s ' % (sys.argv[1]))
    dr = data_reader(sys.argv[1],sys.argv[2])
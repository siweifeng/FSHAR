import os

## data preprocessing for opp
#for i in range(4):
#    os.system('python data_read.py opp '+str(i+1))
#    os.system('python data_preproc_rnn.py opp '+str(i+1))
#    os.system('python data_balcls.py opp '+str(i+1))
#    os.system('python src_tgt_split.py opp '+str(i+1))
#    
## data preprocessing for pamap2
#for i in range(3):
#    os.system('python data_read.py pamap2 '+str(i+1))
#    os.system('python data_preproc_rnn.py pamap2 '+str(i+1))
#    os.system('python data_balcls.py pamap2 '+str(i+1))
#    os.system('python src_tgt_split.py pamap2 '+str(i+1))
    
os.system('python fs_collect.py')
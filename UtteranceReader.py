'''
Created on Jun 18, 2013

@author: vinyals
'''

from external import htkmfc
import numpy as np

class UtteranceReader:
    def __init__(self,list_files):
        self.list_files = list_files
        self.utt_data = []
        
    def GetHTKSize(self):
        file_htk = htkmfc.HTKFeat_read(self.list_files[0])
        file_htk.readheader()
        return file_htk.veclen
        
    def ReadAllUtterances(self, feat_range=None):
        if feat_range == None:
            feat_range = range(self.GetHTKSize())
        self.feat_range = feat_range
        
        for i in range(len(self.list_files)):
            file_htk = htkmfc.HTKFeat_read(self.list_files[i])
            np_data = file_htk.getall()[:,self.feat_range]
            self.utt_data.append(np_data)    
        
if __name__ == '__main__':
    list_files = ['./data/BABEL_BP_104_85455_20120310_210107_outLine','./data/BABEL_BP_104_85455_20120310_210107_outLine','./data/BABEL_BP_104_85455_20120310_210107_outLine']
    feat_range = [0,1,2,5,6,7,69,74]
    utt_reader = UtteranceReader(list_files)
    print utt_reader.GetHTKSize()
    utt_reader.ReadAllUtterances(feat_range)
    print utt_reader.utt_data
    utt_reader.ReadAllUtterances()
    print utt_reader.utt_data
        
'''The Babel dataset 
'''

import cPickle as pickle
from iceberk import datasets, mpi
import numpy as np
import os
import UtteranceReader
import PostingParser
import Sampler

class BabelDataset(datasets.ImageSet):
    """The Bable dataset
    """
    # some  Babel constants
    
    #def __init__(self, utt_reader,posting_sampler):
    def __init__(self, list_file, feat_range, posting_file, perc_pos):
        '''TODO: Read pieces of utterance from the CSV file instead to save memory. It would be nice to index thse by utt_id (by now I do a map).'''
        super(BabelDataset, self).__init__()
        utt_reader = UtteranceReader.UtteranceReader(list_file)
        utt_reader.ReadAllUtterances(feat_range)
        testParser = PostingParser.PostingParser(posting_file)
        self.posting_sampler = Sampler.Sampler(testParser)
        self.posting_sampler.GetPositive()
        self.posting_sampler.GetNegative()
        self.posting_sampler.SampleData(perc_pos)
        
        self._data_all = None
        self._dim = False
        self._channels = 1
        if mpi.is_root():
            self._data = []
            self._label = []
            self._features = []
            for i in range(len(self.posting_sampler.negative_data)):
                if utt_reader.map_utt_idx.has_key(self.posting_sampler.negative_data[i]['file']):
                    if self.posting_sampler.negative_data[i]['sys_bt'] == '':
                        print 'We found a negative example that was not produced by the system!'
                        exit(0)
                    sys_bt = float(self.posting_sampler.negative_data[i]['sys_bt'])
                    sys_et = float(self.posting_sampler.negative_data[i]['sys_et'])
                    sys_sc = float(self.posting_sampler.negative_data[i]['sys_score'])
                    if(sys_et-sys_bt < 0.2):
                        continue
                    self._data.append(utt_reader.GetUtterance(self.posting_sampler.negative_data[i]['file'],
                                                              sys_bt, sys_et))
                    self._label.append(0)
                    self._features.append(sys_sc)
                else:
                    pass
            for i in range(len(self.posting_sampler.positive_data)):
                if utt_reader.map_utt_idx.has_key(self.posting_sampler.positive_data[i]['file']):
                    if self.posting_sampler.positive_data[i]['sys_bt'] == '':
                        sys_bt = 0
                        sys_et = None
                        sys_sc = -1.0
                        #print self.posting_sampler.positive_data[i]['alignment']
                        continue #Should just ignore these?
                    else:
                        sys_bt = float(self.posting_sampler.positive_data[i]['sys_bt'])
                        sys_et = float(self.posting_sampler.positive_data[i]['sys_et'])
                        sys_sc = float(self.posting_sampler.positive_data[i]['sys_score'])
                        if(sys_et-sys_bt < 0.2):
                            continue
                    self._data.append(utt_reader.GetUtterance(self.posting_sampler.positive_data[i]['file'],
                                                              sys_bt, sys_et))
                    self._label.append(1)
                    self._features.append(sys_sc)
                else:
                    pass
            
            self._label = np.array(self._label)
        else:
            self._data = None
            self._label = None
            self._features = None
        self._data = mpi.distribute_list(self._data)
        self._label = mpi.distribute(self._label)
        self._features = mpi.distribute_list(self._features)
        
    def ConvertFeatures(self,feat_range):
        '''Saves a copy for _data (all features), and strips out some features'''
        if self._data_all == None:
            self._data_all = self._data #make a copy the first time
        self._data = []
        for i in range(len(self._data_all)):
            self._data.append(self._data_all[i][:,feat_range])
            
if __name__ == '__main__':
    list_file = './data/list_files.scp'
    feat_range = [0,1,2,5,6,7,69,74]
    posting_file = './data/word.kwlist.alignment.csv'
    perc_pos = 0.2
    babel = BabelDataset(list_file, None, posting_file, perc_pos)
    print babel._data[0].shape
    babel.ConvertFeatures([0,1,3])
    print babel._data[0].shape
    babel.ConvertFeatures([0,1,74])
    print babel._data[0].shape
    babel.ConvertFeatures(range(30))
    print babel._data[0].shape
            
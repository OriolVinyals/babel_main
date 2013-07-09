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
    def __init__(self, list_file, feat_range, posting_file, perc_pos, keep_full_utt=False, posting_sampler=None):
        '''TODO: Read pieces of utterance from the CSV file instead to save memory. It would be nice to index thse by utt_id (by now I do a map).'''
        super(BabelDataset, self).__init__()
        utt_reader = UtteranceReader.UtteranceReader(list_file)
        utt_reader.ReadAllUtterances(feat_range)
        if posting_sampler == None:
            testParser = PostingParser.PostingParser(posting_file)
            self.posting_sampler = Sampler.Sampler(testParser)
            self.posting_sampler.GetPositive()
            self.posting_sampler.GetNegative()
            self.posting_sampler.SampleData(perc_pos)
        else:
            self.posting_sampler = posting_sampler
        
        self._data_all = None
        self._dim = False
        self._channels = 1
        self.keep_full_utt = keep_full_utt
        if mpi.is_root():
            self._data = []
            self._label = []
            self._features = []
            self._utt_id = []
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
                    self._utt_id.append(self.posting_sampler.negative_data[i]['file'])
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
                    self._utt_id.append(self.posting_sampler.positive_data[i]['file'])
                else:
                    pass
            
            self._label = np.array(self._label)
        else:
            self._data = None
            self._label = None
            self._features = None
            self._utt_id = None
        self._data = mpi.distribute_list(self._data)
        self._label = mpi.distribute(self._label)
        self._features = mpi.distribute_list(self._features)
        self._utt_id = mpi.distribute_list(self._utt_id)
        if self.keep_full_utt == True:
            self.utt_reader = utt_reader
        
    def ConvertFeatures(self,feat_range):
        '''Saves a copy for _data (all features), and strips out some features'''
        if self._data_all == None:
            self._data_all = self._data #make a copy the first time
        self._data = []
        for i in range(len(self._data_all)):
            self._data.append(self._data_all[i][:,feat_range])
            
    def GetLocalFeatures(self, feat_type=['entropy','entropy']):
        '''Computes local features. Each mpi node computes its own'''
        self._local_features = []
        for i in range(len(self._data)):
            vector_return = []
            for j in range(len(feat_type)):
                if feat_type[j] == 'entropy':
                    aux = self._data[i]*np.log(self._data[i])
                    aux = np.sum(aux,1)
                    vector_return.append(np.average(aux))
            self._local_features.append(vector_return)
            
    def GetGlobalFeatures(self, feat_type=['entropy','entropy']):
        '''Computes global features. Root computes on all utterances and then mpi distributes to nodes (not sure if the other way around is more efficient)'''
        if self.keep_full_utt == False:
            print 'Error, we need to keep full utterance to compute global (per utterance) features!'
            exit(0)
        if mpi.is_root():
            self._glob_features = []
            for i in range(len(self._data)):
                self._glob_features.append(self.utt_reader.GetGlobFeature(self._utt_id[i], feat_type=feat_type))
        else:
            self._glob_features = None
        self._glob_features = mpi.distribute_list(self._glob_features)
            
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
    print babel._utt_id[0]
            
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
        
        self._data = []
        self._label = []
        self._dim = False
        self._channels = 1
        for i in range(len(self.posting_sampler.negative_data)):
            if utt_reader.map_utt_idx.has_key(self.posting_sampler.negative_data[i]['file']):
                if self.posting_sampler.negative_data[i]['sys_bt'] == None:
                    print 'We found a negative example that was not produced by the system!'
                    exit(0)
                sys_bt = float(self.posting_sampler.negative_data[i]['sys_bt'])
                sys_et = float(self.posting_sampler.negative_data[i]['sys_et'])
                if(sys_et-sys_bt < 0.2):
                    continue
                self._data.append(utt_reader.GetUtterance(self.posting_sampler.negative_data[i]['file'],
                                                          sys_bt, sys_et))
                self._label.append(0)
            else:
                pass
        for i in range(len(self.posting_sampler.positive_data)):
            if utt_reader.map_utt_idx.has_key(self.posting_sampler.positive_data[i]['file']):
                if self.posting_sampler.positive_data[i]['sys_bt'] == '':
                    sys_bt = 0
                    sys_et = None
                    print self.posting_sampler.positive_data[i]['alignment']
                else:
                    sys_bt = float(self.posting_sampler.positive_data[i]['sys_bt'])
                    sys_et = float(self.posting_sampler.positive_data[i]['sys_et'])
                    if(sys_et-sys_bt < 0.2):
                        continue
                self._data.append(utt_reader.GetUtterance(self.posting_sampler.positive_data[i]['file'],
                                                          sys_bt, sys_et))
                self._label.append(1)
            else:
                pass
        
        self._label = np.array(self._label)
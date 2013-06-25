'''The Babel dataset 
'''

import cPickle as pickle
from iceberk import datasets, mpi
import numpy as np
import os
import UtteranceReader
import PostingParser

class BabelDataset(datasets.ImageSet):
    """The Bable dataset
    """
    # some  Babel constants
    
    def __init__(self, utt_reader,posting_sampler):
        '''TODO: Read pieces of utterance from the CSV file instead to save memory. It would be nice to index thse by utt_id (by now I do a map).'''
        super(BabelDataset, self).__init__()
        self._data = utt_reader.utt_data
        self._data = []
        self._dim = False
        self._channels = 1
        for i in range(len(posting_sampler.negative_data)):
            if utt_reader.map_utt_idx.has_key(posting_sampler.negative_data[i]['file']):
                self._data.append(utt_reader.GetUtterance(posting_sampler.negative_data[i]['file'],0,1))
            else:
                pass
        print 'muja'
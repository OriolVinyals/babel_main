'''The Babel dataset 
'''

import cPickle as pickle
from iceberk import datasets, mpi
import numpy as np
import os
import UtteranceReader

class BabelDataset(datasets.ImageSet):
    """The Bable dataset
    """
    # some  Babel constants
    
    def __init__(self, utt_reader):
        super(BabelDataset, self).__init__()
        self._data = utt_reader.utt_data
        self._dim = False
        self._channels = 1
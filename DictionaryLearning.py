import cPickle as pickle
import cProfile
import gflags
import logging
from iceberk import mpi, pipeline
import numpy as np
import os
import sys
import BabelDataset
import UtteranceReader
import PostingParser
import Sampler

if __name__ == '__main__':
    mpi.log_level(logging.DEBUG)
    logging.info('Loading Babel data...')
    list_files = ['./data/BABEL_BP_104_85455_20120310_210107_outLine','./data/BABEL_BP_104_85455_20120310_210107_outLine','./data/BABEL_BP_104_85455_20120310_210107_outLine']
    feat_range = [0,1,2,5,6,7,69,74]
    utt_reader = UtteranceReader.UtteranceReader(list_files)
    utt_reader.ReadAllUtterances(feat_range)
    testParser = PostingParser.PostingParser('./data/word.kwlist.alignment.csv')
    sampler = Sampler.Sampler(testParser)
    sampler.GetPositive()
    sampler.GetNegative()
    sampler.SampleData(0.2)
    
    babel = BabelDataset.BabelDataset(utt_reader,sampler)
    
    conv = pipeline.ConvLayer([
                pipeline.PatchExtractor([6,6], 1), # extracts patches
                pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
                pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
                pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                    trainer = pipeline.OMPTrainer(
                            {'k': 10, 'max_iter':100})), # does encoding
                ])
    logging.info('Training the pipeline...')
    conv.train(babel, 1000)
    print 'muja'
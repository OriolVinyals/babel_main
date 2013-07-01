import logging
from iceberk import mpi, pipeline, classifier
import numpy as np
import BabelDataset

if __name__ == '__main__':
    '''Loading Data: '''
    print 'Rank of this process is ',mpi.RANK
    #mpi.log_level(logging.DEBUG)
    logging.info('Loading Babel data...')
    list_file = './data/20130307.dev.untightened.scp'
    feat_range = range(15)
    posting_file = './data/word.kwlist.alignment.csv'
    perc_pos = 0.2
    babel = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos)
    
    list_file = './data/20130307.eval.untightened.scp'
    posting_file = './data/eval_part1.alignment.csv'
    perc_pos = 0.2
    babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos)
    
    '''An example audio pipeline to extract features'''
    conv = pipeline.ConvLayer([
                pipeline.PatchExtractor([10,15], 1), # extracts patches
                pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
                pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
                pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                    trainer = pipeline.OMPTrainer(
                            {'k': 50, 'max_iter':100})), # does encoding
                pipeline.SpatialPooler({'grid': (1,1), 'method': 'ave'})
                ])
    logging.info('Training the pipeline...')
    conv.train(babel, 10000)
    logging.info('Extracting features...')
    Xp_a1 = conv.process_dataset(babel, as_2d = True)
    
    '''Pipeline that just gets the score'''
    Xp_score = np.asmatrix(babel._features).T
    
    '''Pipeline that cheats'''
    #Xp_cheat = np.asmatrix(babel.labels().astype(np.int)).T

    '''Building appended features'''
    Xtrain = np.hstack((Xp_a1,Xp_score))
    Ytrain = babel.labels().astype(np.int)
    '''Classifier stage'''
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.0)
    accu = np.sum(Ytrain == (np.dot(Xtrain,w)+b).argmax(axis=1).squeeze()) \
            / float(len(Ytrain))
            
    print 'Accuracy is ',accu
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    logging.info('Running Test...')
    Xp_t_a1 = conv.process_dataset(babel_eval, as_2d = True)
    Xp_t_score = np.asmatrix(babel_eval._features).T
    Xtest = np.hstack((Xp_t_a1,Xp_t_score))
    Ytest = babel_eval.labels().astype(np.int)
    
    accu = np.sum(Ytest == (np.dot(Xtest,w)+b).argmax(axis=1).squeeze()) \
            / float(len(Ytest))
            
    print 'Test Accuracy is ',accu
    print 'Test Prior is ',np.sum(Ytest==0)/float(len(Ytest))
import logging
from iceberk import mpi, pipeline, classifier
import numpy as np
import BabelDataset

if __name__ == '__main__':
    '''Loading Data: '''
    mpi.log_level(logging.DEBUG)
    logging.info('Loading Babel data...')
    list_file = './data/list_files.scp'
    feat_range = [0,1,2,5,6,7,69,74]
    posting_file = './data/word.kwlist.alignment.csv'
    perc_pos = 0.2
    babel = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True)
    
    '''An example audio pipeline to extract features'''
    conv = pipeline.ConvLayer([
                pipeline.PatchExtractor([10,8], 1), # extracts patches
                pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
                pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
                pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                    trainer = pipeline.OMPTrainer(
                            {'k': 50, 'max_iter':100})), # does encoding
                pipeline.SpatialPooler({'grid': (1,1), 'method': 'ave'})
                ])
    logging.info('Training the pipeline...')
    conv.train(babel, 1000)
    logging.info('Extracting features...')
    Xp_a1 = conv.process_dataset(babel, as_2d = True)
    
    '''Pipeline that just gets the score'''
    Xp_score = np.asmatrix(babel._features).T
    
    '''An example for posterior features'''
    posting_file = './data/word.kwlist.alignment.csv'
    list_file = './data/post_list_files.scp'
    feat_range = None
    babel_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True,posting_sampler=babel.posting_sampler)
    babel_post.GetLocalFeatures(feat_type=['entropy','duration'])
    babel_post.GetGlobalFeatures(feat_type=['entropy','entropy'])
    Xp_entropy = np.asmatrix(babel_post._local_features)
    Xp_entropy_glob = np.asmatrix(babel_post._glob_features)

    
    '''Pipeline that just gets the score'''
    Xp_score = np.asmatrix(babel._features).T
    
    '''Pipeline that cheats'''
    #Xp_cheat = np.asmatrix(babel.labels().astype(np.int)).T

    '''Building appended features'''
    print 'audio: ',Xp_a1.shape
    print 'local entropy: ',Xp_entropy.shape
    print 'global entropy: ',Xp_entropy_glob.shape
    print 'score: ',Xp_score.shape
    Xtrain = np.hstack((Xp_a1,Xp_entropy,Xp_entropy_glob,Xp_score))
    Ytrain = babel.labels().astype(np.int)
    m, std = classifier.feature_meanstd(Xtrain)
    Xtrain -= m
    Xtrain /= std
    '''Classifier stage'''
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.0)
    accu = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain,w)+b)
    #accu2 = np.sum(Ytrain == (np.dot(Xtrain,w)+b).argmax(axis=1).squeeze()) \
    #        / float(len(Ytrain))
            
    print 'Accuracy is ',accu
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
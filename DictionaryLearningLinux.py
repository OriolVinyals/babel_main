import logging
from iceberk import mpi, pipeline, classifier
import numpy as np
import BabelDataset

if __name__ == '__main__':
    '''Loading Data: '''
    print 'Rank of this process is ',mpi.RANK
    mpi.log_level(logging.DEBUG)
    logging.info('Loading Babel data...')
    list_file = './data/20130307.dev.untightened.scp'
    feat_range = None
    posting_file = './data/word.kwlist.alignment.csv'
    perc_pos = 0.2
    babel = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos)
    
    list_file = './data/20130307.dev.post.untightened.scp'
    feat_range = None
    babel_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=babel.posting_sampler)
    
    list_file = './data/20130307.eval.untightened.scp'
    posting_file = './data/eval_part1.alignment.csv'
    perc_pos = 0.2
    babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos)
    
    list_file = './data/20130307.eval.post.untightened.scp'
    feat_range = None
    babel_eval_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=babel_eval.posting_sampler)


    '''An example audio pipeline to extract features'''
    conv = pipeline.ConvLayer([
                pipeline.PatchExtractor([10,75], 1), # extracts patches
                pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
                pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
                pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                    trainer = pipeline.OMPTrainer(
                            {'k': 500, 'max_iter':100})), # does encoding
                pipeline.SpatialPooler({'grid': (1,1), 'method': 'ave'})
                ])
    logging.info('Training the pipeline...')
    conv.train(babel, 100000)
    logging.info('Extracting features...')
    Xp_a1 = conv.process_dataset(babel, as_2d = True)
    
    '''An example for posterior features'''
    babel_post.GetLocalFeatures(feat_type=['entropy'])
    babel_post.GetGlobalFeatures(feat_type=['entropy'])
    Xp_entropy = np.asmatrix(babel_post._local_features)
    Xp_entropy_glob = np.asmatrix(babel_post._glob_features)
    
    '''Pipeline that just gets the score'''
    Xp_score = np.asmatrix(babel._features).T
    
    '''Pipeline that cheats'''
    #Xp_cheat = np.asmatrix(babel.labels().astype(np.int)).T

    '''Building appended features'''
    Xtrain = np.hstack((Xp_a1,Xp_entropy,Xp_score))
    Ytrain = babel.labels().astype(np.int)
    m, std = classifier.feature_meanstd(Xtrain)
    Xtrain -= m
    Xtrain /= std
    '''Classifier stage'''
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.0)
    accu = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain,w)+b)
            
    print 'Accuracy is ',accu
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    logging.info('Running Test...')
    Xp_t_a1 = conv.process_dataset(babel_eval, as_2d = True)
    babel_eval_post.GetLocalFeatures(feat_type=['entropy'])
    babel_eval_post.GetGlobalFeatures(feat_type=['entropy'])
    Xp_t_entropy = np.asmatrix(babel_eval_post._local_features)
    Xp_t_entropy_glob = np.asmatrix(babel_eval_post._glob_features)
    Xp_t_score = np.asmatrix(babel_eval._features).T
    Xtest = np.hstack((Xp_t_a1,Xp_t_entropy,Xp_t_score))
    Ytest = babel_eval.labels().astype(np.int)
    Xtest -= m
    Xtest /= std
    
    accu = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest,w)+b)
            
    print 'Test Accuracy is ',accu
    print 'Test Prior is ',np.sum(Ytest==0)/float(len(Ytest))
    
    '''Building appended features'''
    Xtrain = Xp_score
    m, std = classifier.feature_meanstd(Xtrain)
    Xtrain -= m
    Xtrain /= std
    '''Classifier stage'''
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.0)
    accu = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain,w)+b)
            
    print 'Score only Accuracy is ',accu
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    logging.info('Running Test...')
    Xtest = Xp_t_score
    Xtest -= m
    Xtest /= std
    
    accu = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest,w)+b)
            
    print 'Score only Test Accuracy is ',accu
    print 'Test Prior is ',np.sum(Ytest==0)/float(len(Ytest))
    
    '''Building appended features'''
    Xtrain = Xp_a1
    m, std = classifier.feature_meanstd(Xtrain)
    Xtrain -= m
    Xtrain /= std
    '''Classifier stage'''
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.0)
    accu = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain,w)+b)
            
    print 'Audio only Accuracy is ',accu
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    logging.info('Running Test...')
    Xtest = Xp_t_a1
    Xtest -= m
    Xtest /= std
    
    accu = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest,w)+b)
            
    print 'Audio only Test Accuracy is ',accu
    print 'Test Prior is ',np.sum(Ytest==0)/float(len(Ytest))
    
    '''Building appended features'''
    Xtrain = Xp_entropy
    m, std = classifier.feature_meanstd(Xtrain)
    Xtrain -= m
    Xtrain /= std
    '''Classifier stage'''
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.0)
    accu = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain,w)+b)
            
    print 'Entropy only Accuracy is ',accu
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    logging.info('Running Test...')
    Xtest = Xp_t_entropy
    Xtest -= m
    Xtest /= std
    
    accu = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest,w)+b)
            
    print 'Entropy only Test Accuracy is ',accu
    print 'Test Prior is ',np.sum(Ytest==0)/float(len(Ytest))
    
    '''Building appended features'''
    Xtrain = Xp_entropy_glob
    m, std = classifier.feature_meanstd(Xtrain)
    Xtrain -= m
    Xtrain /= std
    '''Classifier stage'''
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.0)
    accu = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain,w)+b)
            
    print 'Entropy only Accuracy is ',accu
    print 'Prior Accuracy is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    logging.info('Running Test...')
    Xtest = Xp_t_entropy_glob
    Xtest -= m
    Xtest /= std
    
    accu = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest,w)+b)
            
    print 'Global Entropy only Test Accuracy is ',accu
    print 'Test Prior Accuracy is ',np.sum(Ytest==0)/float(len(Ytest))
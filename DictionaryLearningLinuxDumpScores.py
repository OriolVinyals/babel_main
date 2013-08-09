import logging
from iceberk import mpi, pipeline, classifier
import numpy as np
import BabelDataset
import Classifier
import kws_scorer
import gflags
import sys

gflags.DEFINE_float("perc_pos", 0.2,
                     "Percentage of positive examples to keep.")
FLAGS = gflags.FLAGS


def run():
    '''Loading Data: '''
    print 'Rank of this process is ',mpi.RANK
    mpi.root_log_level(logging.DEBUG)
    logging.info('Loading Babel data...')
    
########### GRID SEARCH SETUP ###########
    
    #GENERAL
    #perc_pos ***CMD
    #feature_set ***CACHE
    #svm/logreg (can't do without other threshold)
    #reg ***CACHE
    
    #ACOUSTIC
    #dict size ***CMD
    #normalization ***CMD
    #alpha ***CMD
    #patchsize ***CMD
    #pool_method **CMD
    
########### TRAIN ###########
    
    perc_pos = FLAGS.perc_pos
    min_dur = 0.2
    posting_sampler = None
    feat_range = None
    Xtrain_dict = {}
    feat_list = None
    
    acoustic=True
    if(acoustic):
        logging.info('****Acoustic Training****')
        list_file = './data/20130307.dev.untightened.scp'
        feat_range = None
        posting_file = './data/word.kwlist.alignment.csv'
        babel = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler,
                                          reader_type='utterance')
        posting_sampler = babel.posting_sampler
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
        normalizer = True
        zca = True
        if normalizer==False:
            del conv[1]
            if zca==False:
                del conv[1]
        else:
            if zca==False:
                del conv[2]
        logging.info('Training the pipeline...')
        conv.train(babel, 200000)
        logging.info('Extracting features...')
        Xp_acoustic = conv.process_dataset(babel, as_2d = True)
        Xtrain_dict['Acoustic'] = Xp_acoustic
        
    lattice=True
    if(lattice):
        logging.info('****Lattice Training****')
        list_file = './data/lat.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                          posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
        posting_sampler = babel_lat.posting_sampler
        #Xtrain_dict['Lattice'] = 0
    
    posterior=True
    if(posterior):
        logging.info('****Posterior Training****')
        list_file = './data/20130307.dev.post.untightened.scp'
        posting_file = './data/word.kwlist.alignment.csv'
        feat_range = None
        babel_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True,reader_type='utterance', 
                                               posting_sampler=posting_sampler,min_dur=min_dur)
        posting_sampler = babel_post.posting_sampler
        #reassign utterances (hack because the scp files are wrong)
        babel_post.utt_reader.list_times_utt = babel_lat.utt_reader.list_times_utt
        '''An example for posterior features'''
        babel_post.GetGlobalFeatures(feat_type=['entropy'])
        babel_post.GetUtteranceFeatures(feat_type=['entropy'])
        Xp_post_glob = np.asmatrix(babel_post._glob_features)
        Xp_post_utt = np.asmatrix(babel_post._utt_features)
        Xtrain_dict['Posterior_Global'] = Xp_post_glob
        Xtrain_dict['Posterior_Utt'] = Xp_post_utt
        
    srate=True
    if(srate):
        logging.info('****Srate Training****')
        list_file = './data/audio.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_srate = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='srate',pickle_fname='./pickles/full.srate.pickle',
                                   posting_sampler=posting_sampler,min_dur=min_dur)
        posting_sampler = babel_srate.posting_sampler
        babel_srate.GetUtteranceFeatures(feat_type=['srate'])
        babel_srate.GetGlobalFeatures(feat_type=['srate'])
        Xp_srate_glob=np.asmatrix(babel_srate._glob_features)
        Xp_srate_utt=np.asmatrix(babel_srate._utt_features)
        Xtrain_dict['Srate_Global'] = Xp_srate_glob.T
        Xtrain_dict['Srate_Utt'] = Xp_srate_utt.T
        
    snr=True
    if(snr):
        logging.info('****SNR Training****')
        list_file = './data/audio.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_snr = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='snr',pickle_fname='./pickles/full.snr.pickle',
                                 posting_sampler=posting_sampler,min_dur=min_dur)
        posting_sampler = babel_snr.posting_sampler
        babel_snr.GetUtteranceFeatures(feat_type=['snr'])
        babel_snr.GetGlobalFeatures(feat_type=['snr'])
        Xp_snr_glob=np.asmatrix(babel_snr._glob_features)
        Xp_snr_utt=np.asmatrix(babel_snr._utt_features)
        Xtrain_dict['SNR_Global'] = Xp_snr_glob.T
        Xtrain_dict['SNR_Utt'] = Xp_snr_utt.T
        
    score=True
    if(score):
        logging.info('****Score Training****')
        list_file = './data/word.kwlist.raw.xml'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                 posting_sampler=posting_sampler,min_dur=min_dur)
        posting_sampler = babel_score.posting_sampler
        babel_score.GetLocalFeatures(feat_type=['raw'])
        Xp_score_local=np.asmatrix(babel_score._local_features)
        Xtrain_dict['Score_Local'] = Xp_score_local

    '''Labels''' 
    feat_list= Xtrain_dict.keys()   
    print 'Features: ' + ' '.join(feat_list)
    for feat in feat_list:
        print feat,Xtrain_dict[feat].shape
    Ytrain = babel_score.labels().astype(np.int)
    
    correlation=False
    if(correlation):
        print np.corrcoef(Ytrain, Xp_score_local.T)
        print np.corrcoef(Ytrain, Xp_snr_glob)
        print np.corrcoef(Ytrain, Xp_snr_utt)
        print np.corrcoef(Ytrain, Xp_srate_glob)
        print np.corrcoef(Ytrain, Xp_srate_utt)
        exit(0)

########### EVAL ###########

    perc_pos = 0.0
    min_dur = 0.2
    posting_sampler = None
    feat_range = None
    Xtest_dict = {}

    eval=True
    if(eval):
        if(acoustic):
            logging.info('****Acoustic Testing****')
            list_file = './data/20130307.eval.untightened.scp'
            feat_range = None
            posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
            babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler)
            posting_sampler = babel_eval.posting_sampler
            '''An example audio pipeline to extract features'''
            Xp_eval_acoustic = conv.process_dataset(babel_eval, as_2d = True)
            Xtest_dict['Acoustic'] = Xp_eval_acoustic
            
        if(lattice):
            logging.info('****Lattice Testing****')
            list_file = './data/lat.eval.list'
            posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
            babel_eval_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                              posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
            posting_sampler = babel_eval_lat.posting_sampler
            Xtest_dict['Lattice'] = 0
        
        if(posterior):
            logging.info('****Posterior Testing****')
            list_file = './data/20130307.eval.post.untightened.scp'
            posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
            feat_range = None
            babel_eval_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=posting_sampler,min_dur=min_dur)
            posting_sampler = babel_eval_post.posting_sampler
            #reassign utterances (hack because the scp files are wrong)
            babel_eval_post.utt_reader.list_times_utt = babel_eval_lat.utt_reader.list_times_utt
            '''An example for posterior features'''
            babel_eval_post.GetGlobalFeatures(feat_type=['entropy'])
            babel_eval_post.GetUtteranceFeatures(feat_type=['entropy'])
            Xp_eval_post_glob = np.asmatrix(babel_eval_post._glob_features)
            Xp_eval_post_utt = np.asmatrix(babel_eval_post._utt_features)
            Xtest_dict['Posterior_Global'] = Xp_eval_post_glob
            Xtest_dict['Posterior_Utt'] = Xp_eval_post_utt
            
        if(srate):
            logging.info('****Srate Testing****')
            list_file = './data/audio.eval.list'
            posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
            babel_eval_srate = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='srate',
                                       pickle_fname='./pickles/full.eval.srate.pickle', posting_sampler=posting_sampler,min_dur=min_dur)
            posting_sampler = babel_eval_srate.posting_sampler
            babel_eval_srate.GetUtteranceFeatures(feat_type=['srate'])
            babel_eval_srate.GetGlobalFeatures(feat_type=['srate'])
            Xp_eval_srate_glob=np.asmatrix(babel_eval_srate._glob_features)
            Xp_eval_srate_utt=np.asmatrix(babel_eval_srate._utt_features)
            Xtest_dict['Srate_Global'] = Xp_eval_srate_glob.T
            Xtest_dict['Srate_Utt'] = Xp_eval_srate_utt.T
            
        if(snr):
            logging.info('****SNR Testing****')
            list_file = './data/audio.eval.list'
            posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
            babel_eval_snr = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='snr',
                                     pickle_fname='./pickles/full.eval.snr.pickle', posting_sampler=posting_sampler,min_dur=min_dur)
            posting_sampler = babel_eval_snr.posting_sampler
            babel_eval_snr.GetUtteranceFeatures(feat_type=['snr'])
            babel_eval_snr.GetGlobalFeatures(feat_type=['snr'])
            Xp_eval_snr_glob=np.asmatrix(babel_eval_snr._glob_features)
            Xp_eval_snr_utt=np.asmatrix(babel_eval_snr._utt_features)
            Xtest_dict['SNR_Global'] = Xp_eval_snr_glob.T
            Xtest_dict['SNR_Utt'] = Xp_eval_snr_utt.T
            
        if(score):
            logging.info('****Score Testing****')
            list_file = './data/word.cut_down_evalpart1.kwlist.raw.xml'
            posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
            babel_eval_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                     posting_sampler=posting_sampler,min_dur=min_dur)
            posting_sampler = babel_eval_score.posting_sampler
            babel_eval_score.GetLocalFeatures(feat_type=['raw'])
            Xp_eval_score_local=np.asmatrix(babel_eval_score._local_features)
            Xtest_dict['Score_Local'] = Xp_eval_score_local

        Ytest = babel_eval_score.labels().astype(np.int)

########### CLASSIFIER ###########

    lr_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    '''Classifier stage'''
    #feat_list=['Local','Utterance']
    w, b = lr_classifier.Train(feat_list=feat_list,type='logreg',gamma=0.0)
    accu = lr_classifier.Accuracy(Xtrain_dict, Ytrain)
    neg_ll = lr_classifier.loss_multiclass_logreg(Xtrain_dict, Ytrain)

    print 'Accuracy is ',accu
    print 'Neg LogLikelihood is ',neg_ll
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    logging.info('Running Test...')
    accu = lr_classifier.Accuracy(Xtest_dict, Ytest)
    neg_ll = lr_classifier.loss_multiclass_logreg(Xtest_dict, Ytest)
    prob = lr_classifier.get_predictions_logreg(Xtest_dict)
      
    print 'Test Accuracy is ',accu
    print 'Test Neg LogLikelihood is ',neg_ll
    print 'Test Prior is ',np.sum(Ytest==0)/float(len(Ytest))
    
    sys_name = './data/eval.'+''.join(feat_list)+'.xml'
    baseline_name = './data/eval.rawscore.xml'
    
    babel_eval_score.DumpScoresXML(sys_name,prob[:,1])
    babel_eval_score.DumpScoresXML(baseline_name,np.asarray(Xp_eval_score_local).squeeze())
    
    print 'ATWV system:',kws_scorer.get_score(sys_name)
    print 'ATWV baseline:',kws_scorer.get_score(baseline_name)
    
if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    run()

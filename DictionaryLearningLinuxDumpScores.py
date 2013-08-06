import logging
from iceberk import mpi, pipeline, classifier
import numpy as np
import BabelDataset
import Classifier
import kws_scorer

if __name__ == '__main__':
    '''Loading Data: '''
    print 'Rank of this process is ',mpi.RANK
    mpi.root_log_level(logging.DEBUG)
    logging.info('Loading Babel data...')
    
########### TRAIN ###########
    
    perc_pos = 0.0
    min_dur = 0.2
    posting_sampler = None
    feat_range = None
    Xtrain_dict = {}
    
    acoustic=False
    if(acoustic):
        list_file = './data/20130307.dev.untightened.scp'
        feat_range = None
        posting_file = './data/word.kwlist.alignment.csv'
        babel = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler)
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
        logging.info('Training the pipeline...')
        conv.train(babel, 100000)
        logging.info('Extracting features...')
        Xp_acoustic = conv.process_dataset(babel, as_2d = True)
        Xtrain_dict['Acoustic'] = Xp_acoustic
        
    lattice=False
    if(lattice):
        list_file = './data/lat.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                          posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
        posting_sampler = babel_lat.posting_sampler
        Xtrain_dict['Lattice'] = 0
    
    posterior=False
    if(posterior):
        list_file = './data/20130307.dev.post.untightened.scp'
        posting_file = './data/word.kwlist.alignment.csv'
        feat_range = None
        babel_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=posting_sampler,min_dur=min_dur)
        posting_sampler = babel_post.posting_sampler
        #reassign utterances (hack because the scp files are wrong)
        babel_post.utt_reader.list_times_utt = babel_lat.utt_reader.list_times_utt
        '''An example for posterior features'''
        logging.info('Getting Local Features')
        babel_post.GetLocalFeatures(feat_type=['score'],fname_xml='./data/word.kwlist.raw.xml')
        logging.info('Getting Global Features')
        babel_post.GetGlobalFeatures(feat_type=['entropy'])
        logging.info('Getting Utterance Features')
        babel_post.GetUtteranceFeatures(feat_type=['entropy'])
        Xp_post_local = np.asmatrix(babel_post._local_features)
        Xp_post_glob = np.asmatrix(babel_post._glob_features)
        Xp_post_utt = np.asmatrix(babel_post._utt_features)
        Xtrain_dict['Posterior_Local'] = Xp_post_local
        Xtrain_dict['Posterior_Global'] = Xp_post_glob
        Xtrain_dict['Posterior_Utt'] = Xp_post_utt
        
    srate=True
    if(srate):
        list_file = './data/audio.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_srate = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='srate',pickle_fname='./pickles/full.srate.pickle',
                                   posting_sampler=posting_sampler,min_dur=min_dur)
        posting_sampler = babel_srate.posting_sampler
        babel_srate.GetUtteranceFeatures('srate')
        babel_srate.GetGlobalFeatures('srate')
        babel_srate.GetLocalFeatures(feat_type=['score'],fname_xml='./data/word.kwlist.raw.xml')
        Xp_srate_glob=np.asmatrix(babel_srate._glob_features)
        Xp_srate_utt=np.asmatrix(babel_srate._utt_features)
        Xp_score_local=np.asmatrix(babel_srate._local_features)
        Xtrain_dict['Srate_Global'] = Xp_srate_glob
        Xtrain_dict['Srate_Utt'] = Xp_srate_utt
        Xtrain_dict['Score_Local'] = Xp_score_local #May want to put this in a different class so I can do score utterance and global features
        
    snr=True
    if(snr):
        list_file = './data/audio.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_snr = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='snr',pickle_fname='./pickles/full.snr.pickle',
                                 posting_sampler=posting_sampler,min_dur=min_dur)
        posting_sampler = babel_snr.posting_sampler
        babel_snr.GetUtteranceFeatures('snr')
        babel_snr.GetGlobalFeatures('snr')
        Xp_snr_glob=np.asmatrix(babel_snr._glob_features)
        Xp_snr_utt=np.asmatrix(babel_snr._utt_features)
        Xtrain_dict['SNR_Global'] = Xp_snr_glob
        Xtrain_dict['SNR_Utt'] = Xp_snr_utt

    '''Labels'''    
    Ytrain = babel_srate.labels().astype(np.int)
    
    correlation=True
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

    eval=False
    if(eval):
        list_file = './data/20130307.eval.untightened.scp'
        posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
        babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur)
        
        list_file = './data/20130307.eval.post.untightened.scp'
        feat_range = None
        babel_eval_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=babel_eval.posting_sampler,min_dur=min_dur)
        
        list_file = './data/lat.eval.list'
        babel_eval_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                                   posting_sampler=babel_eval.posting_sampler,min_dur=min_dur,reader_type='lattice')
        
        #reassign utterances
        babel_eval_post.utt_reader.list_times_utt = babel_eval_lat.utt_reader.list_times_utt
        
#     Xp_t_a1 = conv.process_dataset(babel_eval, as_2d = True)
    logging.info('Getting Eval Local Features')
    babel_eval_post.GetLocalFeatures(feat_type=['score'],fname_xml='./data/word.cut_down_evalpart1.kwlist.raw.xml')
#     logging.info('Getting Eval Global Features')
#     babel_eval_post.GetGlobalFeatures(feat_type=['entropy'])
    logging.info('Getting Eval Utterance Features')
    babel_eval_post.GetUtteranceFeatures(feat_type=['entropy'])
    Xp_t_entropy = np.asmatrix(babel_eval_post._local_features)
#     Xp_t_entropy_glob = np.asmatrix(babel_eval_post._glob_features)
    Xp_t_entropy_utt = np.asmatrix(babel_eval_post._utt_features)
#     Xtest_dict = {'Audio':Xp_t_a1, 'Local':Xp_t_entropy, 'Global':Xp_t_entropy_glob, 'Score':Xp_t_score, 'Utterance':Xp_t_entropy_utt}
    Xtest_dict = {'Local':Xp_t_entropy, 'Utterance':Xp_t_entropy_utt}
    Ytest = babel_eval.labels().astype(np.int)

########### CLASSIFIER ###########

    lr_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    '''Classifier stage'''
    feat_list=['Local','Utterance']
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
    
    babel_eval.DumpScoresXML('./data/eval.localutt.xml',prob[:,1])
    babel_eval.DumpScoresXML('./data/eval.rawscore.xml',np.asarray(Xp_t_entropy).squeeze())
    
    kws_scorer.get_score('./data/eval.localutt.xml')
    kws_scorer.get_score('./data/eval.rawscore.xml')
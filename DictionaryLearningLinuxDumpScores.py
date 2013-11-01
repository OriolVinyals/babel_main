import logging
from iceberk import mpi, pipeline, classifier
import numpy as np
import BabelDataset
import Classifier
import kws_scorer
import gflags
import sys
import cPickle as pickle
import scipy.io as sio
from external import pyroc

gflags.DEFINE_float("perc_pos", 0.0,
                     "Percentage of positive examples to keep.")
gflags.DEFINE_float("min_dur", 0.2,
                     "Minimum duration under which we don't consider the utterance.")

gflags.DEFINE_string("posting_train","./data/word.kwlist.alignment.csv",
                     "Posting list of training data (typically the dev set in Babel)")
gflags.DEFINE_string("list_audio_train","./data/audio.list",
                     "List of audio data of training data (typically the dev set in Babel)")
gflags.DEFINE_string("list_lattice_train","./data/lat.list",
                     "List of lattice data of training data (typically the dev set in Babel)")
#Note about audio / lattice: lattice list contains accurate utterance boundaries and correspondent lattices. 
#Thus, audio list is just  copying the name of utterance file (with the boundaries), with the correct audio diroctories,
#and the class deals with removing / recording boundaries (as some audio features are utterance based) and changing extension to .sph
#Audio list is used in the score reader as well to (solely) determine utterance boundaries / names.
gflags.DEFINE_string("list_scp_feat_train","./data/20130307.dev.untightened.scp",
                     "List of SCP features (MFCC) of training data (typically the dev set in Babel)")
#Utterance boundaries are bogus for this file (that is why the utt_reader.list_times_utt is replaced with the lattice one)
gflags.DEFINE_string("list_scp_post_train","./data/20130307.dev.post.untightened.scp",
                     "List of SCP features (posteriors) of training data (typically the dev set in Babel)")
#Utterance boundaries are bogus for this file (that is why the utt_reader.list_times_utt is replaced with the lattice one)
gflags.DEFINE_string("list_rawscore_train","./data/word.kwlist.raw.xml",
                     "List of raw scores (from lattices) of training data (typically the dev set in Babel)")


gflags.DEFINE_string("posting_eval","./data/word.cut_down_evalpart1.decision.kwlist.alignment.csv",
                     "Posting list of eval data (typically the eval set in Babel)")
gflags.DEFINE_string("list_audio_eval","./data/audio.eval.list",
                     "List of audio data of eval data (typically the eval set in Babel)")
gflags.DEFINE_string("list_lattice_eval","./data/lat.eval.list",
                     "List of lattice data of eval data (typically the eval set in Babel)")
gflags.DEFINE_string("list_scp_feat_eval","./data/20130307.eval.untightened.scp",
                     "List of SCP features (MFCC) of eval data (typically the eval set in Babel)")
gflags.DEFINE_string("list_scp_post_eval","./data/20130307.eval.post.untightened.scp",
                     "List of SCP features (posteriors) of eval data (typically the eval set in Babel)")
gflags.DEFINE_string("list_rawscore_eval","./data/word.cut_down_evalpart1.kwlist.raw.xml",
                     "List of raw scores (from lattices) of eval data (typically the eval set in Babel)")

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
    min_dur = FLAGS.min_dur
    posting_sampler = None
    kw_feat = 1
    feat_range = None
    Xtrain_dict = {}
    feat_list = None
    Xtrain_special_bias = None
    merge_score_list = None
    merge_score_list_eval = None
    merge_score_list_dev = None
    feat_range_score = None
    min_count = 0.0
    max_count= 1000000.0
    K_factor = 10.0
    
    acoustic=False
    if(acoustic):
        logging.info('****Acoustic Training****')
        list_file = FLAGS.list_scp_feat_train
        feat_range = None
        posting_file = FLAGS.posting_train
        babel = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler,
                                          reader_type='utterance', min_count=min_count, max_count=max_count)
        posting_sampler = babel.posting_sampler
        '''An example audio pipeline to extract features'''
        conv = pipeline.ConvLayer([
                    pipeline.PatchExtractor([10,75], 1), # extracts patches
                    pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
                    pipeline.LinearEncoder({},
                    trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
                    pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                        trainer = pipeline.OMPTrainer(
                                {'k': 50, 'max_iter':100})), # does encoding
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
        
    lattice=False
    if(lattice):
        logging.info('****Lattice Training****')
        list_file = FLAGS.list_lattice_train
        posting_file = FLAGS.posting_train
        babel_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                          posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice',min_count=min_count, max_count=max_count)
        posting_sampler = babel_lat.posting_sampler
        #Xtrain_dict['Lattice'] = 0
    
    posterior=False
    if(posterior):
        logging.info('****Posterior Training****')
        list_file = FLAGS.list_scp_post_train
        posting_file = FLAGS.posting_train
        feat_range = None
        babel_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True,reader_type='utterance', 
                                               posting_sampler=posting_sampler,min_dur=min_dur,min_count=min_count, max_count=max_count)
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
        
    srate=False
    if(srate):
        logging.info('****Srate Training****')
        list_file = FLAGS.list_audio_train
        posting_file = FLAGS.posting_train
        babel_srate = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='srate',pickle_fname='./pickles/full.srate.pickle',
                                   posting_sampler=posting_sampler,min_dur=min_dur,min_count=min_count, max_count=max_count)
        posting_sampler = babel_srate.posting_sampler
        babel_srate.GetUtteranceFeatures(feat_type=['srate'])
        babel_srate.GetGlobalFeatures(feat_type=['srate'])
        Xp_srate_glob=np.asmatrix(babel_srate._glob_features)
        Xp_srate_utt=np.asmatrix(babel_srate._utt_features)
        Xtrain_dict['Srate_Global'] = Xp_srate_glob.T
        Xtrain_dict['Srate_Utt'] = Xp_srate_utt.T
        
    snr=False
    if(snr):
        logging.info('****SNR Training****')
        list_file = FLAGS.list_audio_train
        posting_file = FLAGS.posting_train
        babel_snr = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='snr',pickle_fname='./pickles/full.snr.pickle',
                                 posting_sampler=posting_sampler,min_dur=min_dur,min_count=min_count, max_count=max_count)
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
        list_file = FLAGS.list_rawscore_train
        list_file_sph = FLAGS.list_audio_train
        posting_file = FLAGS.posting_train
        #merge_score_list = ['./data/vietnamese_raw_baseline.xml', './data/vietnamese_raw_adambday.xml']
        #merge_score_list_eval = ['./data/vietnamese_raw_baseline.xml', './data/vietnamese_raw_adambday.xml'] #super hack
        #merge_score_list_dev = ['./data/vietnamese_raw_baseline.xml', './data/vietnamese_raw_adambday.xml'] #super hack
        babel_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                 posting_sampler=posting_sampler,min_dur=min_dur,list_file_sph=list_file_sph, 
                                 kw_feat=kw_feat,min_count=min_count, max_count=max_count, merge_score_files=merge_score_list)
        kw_feat = babel_score.map_keyword_feat
        posting_sampler = babel_score.posting_sampler
        #feat_type_local_score=['raw','kw_length','kw_freq','kw_freq_fine']
        #feat_type_local_score=['raw','kw_length','kw_freq','kw_freq_fine','kw_true_freq','kw_true_ratio']
        #feat_type_local_score=['raw_log_odd','raw','kw_length','kw_freq','kw_freq_fine','kw_true_freq','kw_true_ratio']
        #feat_type_local_score=['raw']
        feat_type_local_score=['raw_log_odd','kw_n_est_log_odd']
        #feat_type_local_score=['raw_log_odd','raw','kw_length','kw_freq','kw_freq_fine','kw_n_est_log_odd']
        #feat_type_local_score=['raw_log_odd','raw','kw_length','kw_freq','kw_threshold','kw_n_est','duration','kw_n_est_log_odd']
        #feat_type_local_score=['raw','kw_length','kw_freq','kw_n_est','duration','kw_threshold']
        #feat_type_local_score=['raw','kw_threshold']
        #K_factor = 2000.0 #Gotta do this if we don't use logs
        #feat_range_score=[1,2]
        babel_score.GetLocalFeatures(feat_type=feat_type_local_score,feat_range=feat_range_score)
        babel_score.GetGlobalFeatures(feat_type=['avg'])
        babel_score.GetUtteranceFeatures(feat_type=['avg','min','max'])
        Xp_score_local=np.asmatrix(babel_score._local_features)
        Xp_score_glob=np.asmatrix(babel_score._glob_features)
        Xp_score_utt=np.asmatrix(babel_score._utt_features)
        Xtrain_dict['Score_Local'] = Xp_score_local
        #Xtrain_dict['Score_Utt'] = Xp_score_utt
        #Xtrain_dict['Score_Glob'] = Xp_score_glob
        #feat_type_special_bias=['kw_n_est_log_odd']
        #feat_type_special_bias=['threshold']
        #babel_score.GetLocalFeatures(feat_type=feat_type_special_bias)
        #Xtrain_special_bias = -np.asmatrix(babel_score._local_features)
        #babel_score.GetLocalFeatures(feat_type=['kw_n_est'])
        #Xtrain_weight = 1.0 / np.asarray(babel_score._local_features)
        #Xtrain_weight = np.hstack((Xtrain_weight,Xtrain_weight))
        
    cheating=False
    if(cheating):
        logging.info('****Labels (cheating) Training****')
        Xtrain_dict['Cheating'] = np.asmatrix(babel_score.labels().astype(np.int)).T

    '''Labels''' 
    feat_list= Xtrain_dict.keys()
    feat_list.insert(0, feat_list.pop(feat_list.index('Score_Local')))
    print 'Features: ' + ' '.join(feat_list)
    for feat in feat_list:
        print feat,Xtrain_dict[feat].shape
    Ytrain = babel_score.labels().astype(np.int)
    
    #sio.savemat('./pickles/train.mat',{'Xtrain':Xtrain_dict,'Ytrain':Ytrain})
    ###TEMP
    #nn_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    #nn_classifier.Train(feat_list=feat_list,type='nn_debug',gamma=0.0)
    ###TEMP
    
    correlation=True
    if(correlation):
        for feat in feat_list:
            print feat, np.corrcoef(Ytrain, Xtrain_dict[feat].T)[0,1:]
            #if Xtrain_dict.has_key('Score_Local'):
            #    print feat, np.corrcoef(Xtrain_dict['Score_Local'].T, Xtrain_dict[feat].T)[0,1:]
        #exit(0)

########### EVAL ###########

    perc_pos = 0.0
    min_dur = FLAGS.min_dur
    posting_sampler = None
    feat_range = None
    Xtest_dict = {}
    Xtest_special_bias = None

    eval=True
    if(eval):
        posting_file = FLAGS.posting_eval
        if(acoustic):
            logging.info('****Acoustic Testing****')
            list_file = FLAGS.list_scp_feat_eval
            feat_range = None
            babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler)
            posting_sampler = babel_eval.posting_sampler
            '''An example audio pipeline to extract features'''
            Xp_eval_acoustic = conv.process_dataset(babel_eval, as_2d = True)
            Xtest_dict['Acoustic'] = Xp_eval_acoustic
            
        if(lattice):
            logging.info('****Lattice Testing****')
            list_file = FLAGS.list_lattice_eval
            babel_eval_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                              posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
            posting_sampler = babel_eval_lat.posting_sampler
            Xtest_dict['Lattice'] = 0
        
        if(posterior):
            logging.info('****Posterior Testing****')
            list_file = FLAGS.list_scp_post_eval
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
            list_file = FLAGS.list_audio_eval
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
            list_file = FLAGS.list_audio_eval
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
            list_file = FLAGS.list_rawscore_eval
            list_file_sph = FLAGS.list_audio_eval
            #merge_score_list_eval = ['./data/vietnamese_raw_baseline.xml', './data/vietnamese_raw_adambday.xml']
            babel_eval_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                     posting_sampler=posting_sampler,min_dur=min_dur,list_file_sph=list_file_sph,
                                     kw_feat=kw_feat, merge_score_files=merge_score_list_eval)
            posting_sampler = babel_eval_score.posting_sampler
            babel_eval_score.GetLocalFeatures(feat_type=feat_type_local_score,feat_range=feat_range_score)
            babel_eval_score.GetGlobalFeatures(feat_type=['avg'])
            babel_eval_score.GetUtteranceFeatures(feat_type=['avg','min','max'])
            Xp_eval_score_local=np.asmatrix(babel_eval_score._local_features)
            Xp_eval_score_glob=np.asmatrix(babel_eval_score._glob_features)
            Xp_eval_score_utt=np.asmatrix(babel_eval_score._utt_features)
            Xtest_dict['Score_Local'] = Xp_eval_score_local
            Xtest_dict['Score_Utt'] = Xp_eval_score_utt
            Xtest_dict['Score_Glob'] = Xp_eval_score_glob
            #babel_eval_score.GetLocalFeatures(feat_type=feat_type_special_bias)
            #Xtest_special_bias = -np.asmatrix(babel_eval_score._local_features)
            
        if(cheating):
            logging.info('****Labels (cheating) Testing****')
            Xtest_dict['Cheating'] = np.asmatrix(babel_eval_score.labels().astype(np.int)).T


        Ytest = babel_eval_score.labels().astype(np.int)
        #sio.savemat('./pickles/eval.mat',{'Xtest':Xtest_dict,'Ytest':Ytest})

########### DEV ###########

    perc_pos = 0.0
    min_dur = FLAGS.min_dur
    posting_sampler = None
    feat_range = None
    Xdev_dict = {}
    Xdev_special_bias = None

    dev=True
    if(dev):
        posting_file = FLAGS.posting_train
        if(acoustic):
            logging.info('****Acoustic Dev****')
            list_file = FLAGS.list_scp_feat_train
            feat_range = None
            babel_dev = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler)
            posting_sampler = babel_dev.posting_sampler
            '''An example audio pipeline to extract features'''
            Xp_dev_acoustic = conv.process_dataset(babel_dev, as_2d = True)
            Xdev_dict['Acoustic'] = Xp_dev_acoustic
            
        if(lattice):
            logging.info('****Lattice Dev****')
            list_file = FLAGS.list_lattice_train
            babel_dev_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                              posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
            posting_sampler = babel_dev_lat.posting_sampler
            Xdev_dict['Lattice'] = 0
        
        if(posterior):
            logging.info('****Posterior Dev****')
            list_file = FLAGS.list_scp_post_train
            feat_range = None
            babel_dev_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=posting_sampler,min_dur=min_dur)
            posting_sampler = babel_dev_post.posting_sampler
            #reassign utterances (hack because the scp files are wrong)
            babel_dev_post.utt_reader.list_times_utt = babel_dev_lat.utt_reader.list_times_utt
            '''An example for posterior features'''
            babel_dev_post.GetGlobalFeatures(feat_type=['entropy'])
            babel_dev_post.GetUtteranceFeatures(feat_type=['entropy'])
            Xp_dev_post_glob = np.asmatrix(babel_dev_post._glob_features)
            Xp_dev_post_utt = np.asmatrix(babel_dev_post._utt_features)
            Xdev_dict['Posterior_Global'] = Xp_dev_post_glob
            Xdev_dict['Posterior_Utt'] = Xp_dev_post_utt
            
        if(srate):
            logging.info('****Srate Dev****')
            list_file = FLAGS.list_audio_train
            babel_dev_srate = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='srate',
                                       pickle_fname='./pickles/full.srate.pickle', posting_sampler=posting_sampler,min_dur=min_dur)
            posting_sampler = babel_dev_srate.posting_sampler
            babel_dev_srate.GetUtteranceFeatures(feat_type=['srate'])
            babel_dev_srate.GetGlobalFeatures(feat_type=['srate'])
            Xp_dev_srate_glob=np.asmatrix(babel_dev_srate._glob_features)
            Xp_dev_srate_utt=np.asmatrix(babel_dev_srate._utt_features)
            Xdev_dict['Srate_Global'] = Xp_dev_srate_glob.T
            Xdev_dict['Srate_Utt'] = Xp_dev_srate_utt.T
            
        if(snr):
            logging.info('****SNR Dev****')
            list_file = FLAGS.list_audio_train
            babel_dev_snr = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='snr',
                                     pickle_fname='./pickles/full.snr.pickle', posting_sampler=posting_sampler,min_dur=min_dur)
            posting_sampler = babel_dev_snr.posting_sampler
            babel_dev_snr.GetUtteranceFeatures(feat_type=['snr'])
            babel_dev_snr.GetGlobalFeatures(feat_type=['snr'])
            Xp_dev_snr_glob=np.asmatrix(babel_dev_snr._glob_features)
            Xp_dev_snr_utt=np.asmatrix(babel_dev_snr._utt_features)
            Xdev_dict['SNR_Global'] = Xp_dev_snr_glob.T
            Xdev_dict['SNR_Utt'] = Xp_dev_snr_utt.T
            
        if(score):
            logging.info('****Score Dev****')
            list_file = FLAGS.list_rawscore_train
            list_file_sph = FLAGS.list_audio_train
            babel_dev_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                     posting_sampler=posting_sampler,min_dur=min_dur,list_file_sph=list_file_sph,
                                     kw_feat=kw_feat, merge_score_files=merge_score_list_dev)
            posting_sampler = babel_dev_score.posting_sampler
            babel_dev_score.GetLocalFeatures(feat_type=feat_type_local_score,feat_range=feat_range_score)
            babel_dev_score.GetGlobalFeatures(feat_type=['avg'])
            babel_dev_score.GetUtteranceFeatures(feat_type=['avg','min','max'])
            Xp_dev_score_local=np.asmatrix(babel_dev_score._local_features)
            Xp_dev_score_glob=np.asmatrix(babel_dev_score._glob_features)
            Xp_dev_score_utt=np.asmatrix(babel_dev_score._utt_features)
            Xdev_dict['Score_Local'] = Xp_dev_score_local
            Xdev_dict['Score_Utt'] = Xp_dev_score_utt
            Xdev_dict['Score_Glob'] = Xp_dev_score_glob
            #babel_dev_score.GetLocalFeatures(feat_type=feat_type_special_bias)
            #Xdev_special_bias = -np.asmatrix(babel_dev_score._local_features)
            
        if(cheating):
            logging.info('****Labels (cheating) Dev****')
            Xdev_dict['Cheating'] = np.asmatrix(babel_dev_score.labels().astype(np.int)).T

        Ydev = babel_dev_score.labels().astype(np.int)

########### CLASSIFIER ###########

    print 'Classifier Stage'
    lr_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    nnet=False
    if nnet:
        nn_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    '''Classifier stage'''
    Xtrain_special_bias=None
    Xdev_special_bias=None
    Xtest_special_bias=None
    lr_classifier.Train(feat_list=feat_list,type='logreg_atwv',gamma=0.000, domeanstd=False, special_bias=Xtrain_special_bias, add_bias=True, 
                        class_instance=babel_dev_score, factor=K_factor, 
                        cv_class_instance=babel_eval_score, cv_feats=Xtest_dict, cv_special_bias=Xtest_special_bias)
    try:
        print lr_classifier.b,lr_classifier.w
    except:
        pass
    if nnet:
        nn_classifier.Train(feat_list=feat_list,type='nn_atwv',gamma=0.000, domeanstd=True, special_bias=Xtrain_special_bias, add_bias=True, 
                            class_instance=babel_dev_score, arch=[40], factor=K_factor,
                            cv_class_instance=babel_eval_score, cv_feats=Xtest_dict, cv_special_bias=Xtest_special_bias)

    accu = lr_classifier.Accuracy(Xtrain_dict, Ytrain, special_bias=Xtrain_special_bias)
    neg_ll = lr_classifier.loss_multiclass_logreg(Xtrain_dict, Ytrain, special_bias=Xtrain_special_bias)
    if nnet:
        accu_nn = nn_classifier.Accuracy(Xtrain_dict, Ytrain)
        neg_ll_nn = nn_classifier.loss_multiclass_nn(Xtrain_dict, Ytrain)

    print 'Accuracy is ',accu
    print 'Neg LogLikelihood is ',neg_ll
    if nnet:
        print 'NN Accuracy is ',accu_nn
        print 'NN Neg LogLikelihood is ',neg_ll_nn
    print 'Prior is ',np.sum(Ytrain==0)/float(len(Ytrain))
    
    if(dev):
        logging.info('Running Dev...')
        accu = lr_classifier.Accuracy(Xdev_dict, Ydev, special_bias=Xdev_special_bias)
        neg_ll = lr_classifier.loss_multiclass_logreg(Xdev_dict, Ydev, special_bias=Xdev_special_bias)
        prob_dev = lr_classifier.get_predictions_logreg(Xdev_dict, special_bias=Xdev_special_bias)
        if nnet:
            accu_nn = nn_classifier.Accuracy(Xdev_dict, Ydev)
            neg_ll_nn = nn_classifier.loss_multiclass_nn(Xdev_dict, Ydev)
            prob_dev_nn = nn_classifier.get_predictions_nn(Xdev_dict)
          
        print 'Dev Accuracy is ',accu
        print 'Dev Neg LogLikelihood is ',neg_ll
        if nnet:
            print 'NN Dev Accuracy is ',accu_nn
            print 'NN Dev Neg LogLikelihood is ',neg_ll_nn
        print 'Dev Prior is ',np.sum(Ydev==0)/float(len(Ydev))
        
        plot_roc=False
        if plot_roc:
            roc_1 = []
            for i in range(len(Ydev)):
                roc_1.append((Ydev[i],prob_dev[i,1]))
            roc_2 = []
            for i in range(len(Ydev)):
                roc_2.append((Ydev[i],Xp_dev_score_local[i,0]))
            if nnet:
                roc_3 = []
                for i in range(len(Ydev)):
                    roc_3.append((Ydev[i],prob_dev_nn[i,1]))
            r1 = pyroc.ROCData(roc_1)
            r2 = pyroc.ROCData(roc_2)
            lista = [r1,r2]
            if nnet:
                r3 = pyroc.ROCData(roc_3)
                lista = [r1,r2,r3]
                pyroc.plot_multiple_roc(lista,'Multiple ROC Curves',labels=['system','baseline','nnet'],include_baseline=True)
            else:
                pyroc.plot_multiple_roc(lista,'Multiple ROC Curves',labels=['system','baseline'],include_baseline=True)
        
        print 'Dev ATWV system:',babel_dev_score.GetATWV(prob_dev[:,1], compute_th=True)
        print 'Dev ATWV no threshold system:',babel_dev_score.GetATWV(prob_dev[:,1])
        if nnet:
            print 'NN Dev ATWV system:',babel_dev_score.GetATWV(prob_dev_nn[:,1], compute_th=True)
            print 'NN Dev ATWV no threshold system:',babel_dev_score.GetATWV(prob_dev_nn[:,1])
        print 'Dev ATWV baseline:',babel_dev_score.GetATWV(np.asarray(Xp_dev_score_local[:,0]).squeeze(),compute_th=True)
        
    logging.info('Running Test...')
    accu = lr_classifier.Accuracy(Xtest_dict, Ytest, special_bias=Xtest_special_bias)
    neg_ll = lr_classifier.loss_multiclass_logreg(Xtest_dict, Ytest, special_bias=Xtest_special_bias)
    prob = lr_classifier.get_predictions_logreg(Xtest_dict, special_bias=Xtest_special_bias)
    if nnet:
        accu_nn = nn_classifier.Accuracy(Xtest_dict, Ytest)
        neg_ll_nn = nn_classifier.loss_multiclass_nn(Xtest_dict, Ytest)
        prob_nn = nn_classifier.get_predictions_nn(Xtest_dict)
      
    print 'Test Accuracy is ',accu
    print 'Test Neg LogLikelihood is ',neg_ll
    if nnet:
        print 'NN Test Accuracy is ',accu_nn
        print 'NN Test Neg LogLikelihood is ',neg_ll_nn
    print 'Test Prior is ',np.sum(Ytest==0)/float(len(Ytest))
    
    print 'ATWV system:',babel_eval_score.GetATWV(prob[:,1], compute_th=True)
    print 'ATWV no threshold system',babel_eval_score.GetATWV(prob[:,1])
    if nnet:
        print 'NN ATWV system:',babel_eval_score.GetATWV(prob_nn[:,1], compute_th=True)
        print 'NN ATWV no threshold system',babel_eval_score.GetATWV(prob_nn[:,1])
    print 'ATWV baseline:',babel_eval_score.GetATWV(np.asarray(Xp_eval_score_local[:,0]).squeeze(),compute_th=True)
    print 'ATWV baseline no th:',babel_eval_score.GetATWV(np.asarray(Xp_eval_score_local[:,0]).squeeze(),compute_th=False)
    
if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    run()

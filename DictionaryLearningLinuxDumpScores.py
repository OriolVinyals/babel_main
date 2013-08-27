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
    kw_feat = 1
    feat_range = None
    Xtrain_dict = {}
    feat_list = None
    Xtrain_special_bias = None
    
    acoustic=False
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
        
    lattice=False
    if(lattice):
        logging.info('****Lattice Training****')
        list_file = './data/lat.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                          posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
        posting_sampler = babel_lat.posting_sampler
        #Xtrain_dict['Lattice'] = 0
    
    posterior=False
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
        
    srate=False
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
        
    snr=False
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
        list_file_sph = './data/audio.list'
        posting_file = './data/word.kwlist.alignment.csv'
        babel_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                 posting_sampler=posting_sampler,min_dur=min_dur,list_file_sph=list_file_sph, 
                                 kw_feat=kw_feat)
        kw_feat = babel_score.map_keyword_feat
        posting_sampler = babel_score.posting_sampler
        #feat_type_local_score=['raw','kw_length','kw_freq','kw_freq_fine']
        #feat_type_local_score=['raw','kw_length','kw_freq','kw_freq_fine','kw_true_freq','kw_true_ratio']
        #feat_type_local_score=['raw','threshold']
        feat_type_local_score=['raw']
        #feat_type_local_score=['raw_log_odd','raw','kew_length','kw_freq','kw_freq_fine','kw_true_freq','kw_true_ratio']
        babel_score.GetLocalFeatures(feat_type=feat_type_local_score)
        babel_score.GetGlobalFeatures(feat_type=['avg'])
        babel_score.GetUtteranceFeatures(feat_type=['avg','min','max'])
        Xp_score_local=np.asmatrix(babel_score._local_features)
        Xp_score_glob=np.asmatrix(babel_score._glob_features)
        Xp_score_utt=np.asmatrix(babel_score._utt_features)
        Xtrain_dict['Score_Local'] = Xp_score_local
        #Xtrain_dict['Score_Utt'] = Xp_score_utt
        #Xtrain_dict['Score_Glob'] = Xp_score_glob
        babel_score.GetLocalFeatures(feat_type=['threshold'])
        Xtrain_special_bias = -np.asmatrix(babel_score._local_features)
        babel_score.GetLocalFeatures(feat_type=['n_est'])
        Xtrain_weight = 1.0 / np.asarray(babel_score._local_features)
        Xtrain_weight = np.hstack((Xtrain_weight,Xtrain_weight))
        
    cheating=False
    if(cheating):
        logging.info('****Labels (cheating) Training****')
        Xtrain_dict['Cheating'] = np.asmatrix(babel_score.labels().astype(np.int)).T

    '''Labels''' 
    feat_list= Xtrain_dict.keys()   
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
    min_dur = 0.2
    posting_sampler = None
    feat_range = None
    Xtest_dict = {}
    Xtest_special_bias = None

    eval=True
    if(eval):
        posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
        if(acoustic):
            logging.info('****Acoustic Testing****')
            list_file = './data/20130307.eval.untightened.scp'
            feat_range = None
            babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler)
            posting_sampler = babel_eval.posting_sampler
            '''An example audio pipeline to extract features'''
            Xp_eval_acoustic = conv.process_dataset(babel_eval, as_2d = True)
            Xtest_dict['Acoustic'] = Xp_eval_acoustic
            
        if(lattice):
            logging.info('****Lattice Testing****')
            list_file = './data/lat.eval.list'
            babel_eval_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                              posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
            posting_sampler = babel_eval_lat.posting_sampler
            Xtest_dict['Lattice'] = 0
        
        if(posterior):
            logging.info('****Posterior Testing****')
            list_file = './data/20130307.eval.post.untightened.scp'
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
            list_file_sph = './data/audio.eval.list'
            babel_eval_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                     posting_sampler=posting_sampler,min_dur=min_dur,list_file_sph=list_file_sph,
                                     kw_feat=kw_feat)
            posting_sampler = babel_eval_score.posting_sampler
            babel_eval_score.GetLocalFeatures(feat_type=feat_type_local_score)
            babel_eval_score.GetGlobalFeatures(feat_type=['avg'])
            babel_eval_score.GetUtteranceFeatures(feat_type=['avg','min','max'])
            Xp_eval_score_local=np.asmatrix(babel_eval_score._local_features)
            Xp_eval_score_glob=np.asmatrix(babel_eval_score._glob_features)
            Xp_eval_score_utt=np.asmatrix(babel_eval_score._utt_features)
            Xtest_dict['Score_Local'] = Xp_eval_score_local
            Xtest_dict['Score_Utt'] = Xp_eval_score_utt
            Xtest_dict['Score_Glob'] = Xp_eval_score_glob
            babel_eval_score.GetLocalFeatures(feat_type=['threshold'])
            Xtest_special_bias = -np.asmatrix(babel_eval_score._local_features)
            
        if(cheating):
            logging.info('****Labels (cheating) Testing****')
            Xtest_dict['Cheating'] = np.asmatrix(babel_eval_score.labels().astype(np.int)).T


        Ytest = babel_eval_score.labels().astype(np.int)
        #sio.savemat('./pickles/eval.mat',{'Xtest':Xtest_dict,'Ytest':Ytest})

########### DEV ###########

    perc_pos = 0.0
    min_dur = 0.2
    posting_sampler = None
    feat_range = None
    Xdev_dict = {}
    Xdev_special_bias = None

    dev=True
    if(dev):
        posting_file = './data/word.kwlist.alignment.csv'
        if(acoustic):
            logging.info('****Acoustic Dev****')
            list_file = './data/20130307.dev.untightened.scp'
            feat_range = None
            babel_dev = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur, posting_sampler=posting_sampler)
            posting_sampler = babel_dev.posting_sampler
            '''An example audio pipeline to extract features'''
            Xp_dev_acoustic = conv.process_dataset(babel_dev, as_2d = True)
            Xdev_dict['Acoustic'] = Xp_dev_acoustic
            
        if(lattice):
            logging.info('****Lattice Dev****')
            list_file = './data/lat.list'
            babel_dev_lat = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, 
                                              posting_sampler=posting_sampler,min_dur=min_dur,reader_type='lattice')
            posting_sampler = babel_dev_lat.posting_sampler
            Xdev_dict['Lattice'] = 0
        
        if(posterior):
            logging.info('****Posterior Dev****')
            list_file = './data/20130307.dev.post.untightened.scp'
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
            list_file = './data/audio.list'
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
            list_file = './data/audio.list'
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
            list_file = './data/word.kwlist.raw.xml'
            list_file_sph = './data/audio.list'
            babel_dev_score = BabelDataset.BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='score',
                                     posting_sampler=posting_sampler,min_dur=min_dur,list_file_sph=list_file_sph,
                                     kw_feat=kw_feat)
            posting_sampler = babel_dev_score.posting_sampler
            babel_dev_score.GetLocalFeatures(feat_type=feat_type_local_score)
            babel_dev_score.GetGlobalFeatures(feat_type=['avg'])
            babel_dev_score.GetUtteranceFeatures(feat_type=['avg','min','max'])
            Xp_dev_score_local=np.asmatrix(babel_dev_score._local_features)
            Xp_dev_score_glob=np.asmatrix(babel_dev_score._glob_features)
            Xp_dev_score_utt=np.asmatrix(babel_dev_score._utt_features)
            Xdev_dict['Score_Local'] = Xp_dev_score_local
            Xdev_dict['Score_Utt'] = Xp_dev_score_utt
            Xdev_dict['Score_Glob'] = Xp_dev_score_glob
            babel_dev_score.GetLocalFeatures(feat_type=['threshold'])
            Xdev_special_bias = -np.asmatrix(babel_dev_score._local_features)
            
        if(cheating):
            logging.info('****Labels (cheating) Dev****')
            Xdev_dict['Cheating'] = np.asmatrix(babel_dev_score.labels().astype(np.int)).T

        Ydev = babel_dev_score.labels().astype(np.int)

########### CLASSIFIER ###########

    lr_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    nnet=False
    if nnet:
        nn_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    '''Classifier stage'''
    #feat_list=['Local','Utterance']
    #Xtrain_special_bias=None
    #Xdev_special_bias=None
    #Xtest_special_bias=None
    lr_classifier.Train(feat_list=feat_list,type='logreg',gamma=0.0, domeanstd=False, special_bias=Xtrain_special_bias, add_bias=False)
    #lr_classifier.Train(feat_list=feat_list,type='linsvm',gamma=0.0, domeanstd=False, add_bias=True)
    print lr_classifier.b,lr_classifier.w
    #lr_classifier.w[0,0]=-1
    #lr_classifier.w[0,1]=1
    if nnet:
        nn_classifier.Train(feat_list=feat_list,type='nn_debug',gamma=0.0)

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
        #a_list = (0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
        #a_list = (1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19)
        #a_list = (0.0,0.0)
        a_list = (0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89)
        best_atwv = 0
        for a in a_list:
            logging.info('Running Dev...')
            print 'A value',a
            lr_classifier.w[0,0]=-a
            lr_classifier.w[0,1]=a
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
            sys_name_dev = './data/dev.'+''.join(feat_list)+'.xml'
            sys_name_dev_nn = './data/dev.'+''.join(feat_list)+'.NN.xml'
            baseline_name_dev = './data/dev.rawscore.xml'
            babel_dev_score.DumpScoresXML(sys_name_dev,prob_dev[:,1])
            if nnet:
                babel_dev_score.DumpScoresXML(sys_name_dev_nn,prob_dev_nn[:,1])
            babel_dev_score.DumpScoresXML(baseline_name_dev,np.asarray(Xp_dev_score_local[:,0]).squeeze())
            
            plot_roc=False
            if plot_roc:
                roc_1 = []
                for i in range(len(Ydev)):
                    roc_1.append((Ydev[i],prob_dev[i,1]))
                roc_2 = []
                for i in range(len(Ydev)):
                    roc_2.append((Ydev[i],Xp_dev_score_local[i,0]))
                r1 = pyroc.ROCData(roc_1)
                r2 = pyroc.ROCData(roc_2)
                lista = [r1,r2]
                pyroc.plot_multiple_roc(lista,'Multiple ROC Curves',labels=['system','baseline'],include_baseline=True)
            
            print 'Dev ATWV system:',kws_scorer.get_score_dev(sys_name_dev)
            atwv=kws_scorer.get_score_woth_dev(sys_name_dev)
            if atwv>best_atwv:
                best_a = a
                best_atwv=atwv
            print 'Dev ATWV no threshold system:',atwv
            if nnet:
                print 'NN Dev ATWV system:',kws_scorer.get_score_dev(sys_name_dev_nn)
                print 'NN Dev ATWV no threshold system:',kws_scorer.get_score_woth_dev(sys_name_dev_nn)
            print 'Dev ATWV baseline:',kws_scorer.get_score_dev(baseline_name_dev)
    
    lr_classifier.w[0,0]=-best_a
    lr_classifier.w[0,1]=best_a
        
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
    
    sys_name = './data/eval.'+''.join(feat_list)+'.xml'
    sys_name_nn = './data/eval.'+''.join(feat_list)+'.NN.xml'
    baseline_name = './data/eval.rawscore.xml'
    
    babel_eval_score.DumpScoresXML(sys_name,prob[:,1])
    if nnet:
        babel_eval_score.DumpScoresXML(sys_name_nn,prob_nn[:,1])
    babel_eval_score.DumpScoresXML(baseline_name,np.asarray(Xp_eval_score_local[:,0]).squeeze())
    
    print 'ATWV system:',kws_scorer.get_score(sys_name)
    print 'ATWV no threshold system',kws_scorer.get_score_woth(sys_name)
    if nnet:
        print 'NN ATWV system:',kws_scorer.get_score(sys_name_nn)
        print 'NN ATWV no threshold system',kws_scorer.get_score_woth(sys_name_nn)
    print 'ATWV baseline:',kws_scorer.get_score(baseline_name)
    
if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    run()

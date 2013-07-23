import logging
from iceberk import mpi, pipeline, classifier
import numpy as np
import BabelDataset
import Classifier
import os

if __name__ == '__main__':
    '''Loading Data: '''
    print 'Rank of this process is ',mpi.RANK
    mpi.root_log_level(logging.DEBUG)
    logging.info('Loading Babel data...')
    list_file = './data/20130307.dev.untightened.scp'
    feat_range = None
    posting_file = './data/word.kwlist.alignment.csv'
    perc_pos = 0.0
    min_dur = 0.2
    babel = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur)
    
    list_file = './data/20130307.dev.post.untightened.scp'
    feat_range = None
    babel_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=babel.posting_sampler,min_dur=min_dur)
    
    list_file = './data/20130307.eval.untightened.scp'
    posting_file = './data/word.cut_down_evalpart1.decision.kwlist.alignment.csv'
    perc_pos = 0.0
    babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur)
    
    list_file = './data/20130307.eval.post.untightened.scp'
    feat_range = None
    babel_eval_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=babel_eval.posting_sampler,min_dur=min_dur)
    
#     '''An example audio pipeline to extract features'''
#     conv = pipeline.ConvLayer([
#                 pipeline.PatchExtractor([10,75], 1), # extracts patches
#                 pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
#                 pipeline.LinearEncoder({},
#                 trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
#                 pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
#                     trainer = pipeline.OMPTrainer(
#                             {'k': 500, 'max_iter':100})), # does encoding
#                 pipeline.SpatialPooler({'grid': (1,1), 'method': 'ave'})
#                 ])
#     logging.info('Training the pipeline...')
#     conv.train(babel, 100000)
#     logging.info('Extracting features...')
#     Xp_a1 = conv.process_dataset(babel, as_2d = True)
    
    '''An example for posterior features'''
    babel_post.GetLocalFeatures(feat_type=['score'],fname_xml='./data/word.kwlist.raw.xml')
#     babel_post.GetGlobalFeatures(feat_type=['entropy'])
#     babel_post.GetUtteranceFeatures(feat_type=['entropy'])
    Xp_entropy = np.asmatrix(babel_post._local_features)
#     Xp_entropy_glob = np.asmatrix(babel_post._glob_features)
#     Xp_entropy_utt = np.asmatrix(babel_post._utt_features)
    
#     '''Pipeline that just gets the score'''
#     Xp_score = np.asmatrix(babel._features).T

    '''Constructing Dictionary of Features'''    
#     Xtrain_dict = {'Audio':Xp_a1, 'Local':Xp_entropy, 'Global':Xp_entropy_glob, 'Score':Xp_score, 'Utterance':Xp_entropy_utt}
    Xtrain_dict = {'Local':Xp_entropy}
    Ytrain = babel.labels().astype(np.int)

#     Xp_t_a1 = conv.process_dataset(babel_eval, as_2d = True)
    babel_eval_post.GetLocalFeatures(feat_type=['score'],fname_xml='./data/word.cut_down_evalpart1.kwlist.raw.xml')
#     babel_eval_post.GetGlobalFeatures(feat_type=['entropy'])
#     babel_eval_post.GetUtteranceFeatures(feat_type=['entropy'])
    Xp_t_entropy = np.asmatrix(babel_eval_post._local_features)
#     Xp_t_entropy_glob = np.asmatrix(babel_eval_post._glob_features)
#     Xp_t_entropy_utt = np.asmatrix(babel_eval_post._utt_features)
#     Xp_t_score = np.asmatrix(babel_eval._features).T
#     Xtest_dict = {'Audio':Xp_t_a1, 'Local':Xp_t_entropy, 'Global':Xp_t_entropy_glob, 'Score':Xp_t_score, 'Utterance':Xp_t_entropy_utt}
    Xtest_dict = {'Local':Xp_t_entropy}
    Ytest = babel_eval.labels().astype(np.int)

    lr_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    '''Classifier stage'''
    feat_list=['Local']
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
    
    babel_eval.DumpScoresXML('./data/eval.local.xml',prob[:,1])
    babel_eval.DumpScoresXML('./data/eval.rawscore.xml',np.asarray(Xp_t_entropy).squeeze())
    
    score_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.local.xml'

    thresh_bin = '/u/swegmann/work/std/kws/score_norm/thresh.pl'
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1.ecf.xml'
    decision_out_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.local.decision.xml'
    cmd = thresh_bin + ' ' + gt_file + ' ' + score_file + ' > ' + decision_out_file
    print 'Running ',cmd
    os.system(cmd)
    
    eval_bin = 'perl5.14.2 /u/drspeech/projects/swordfish/ThirdParty/F4DE/bin/KWSEval'
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.scoring.ecf.xml'
    rttm_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.mitllfa3.rttm'
    t_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-eval.kwlist2.xml'
    opts = '-o -b -O -B -c'
    out_dir = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.local.decision'
    cmd = eval_bin + ' -e ' + gt_file + ' -r ' + rttm_file + ' -t ' + t_file + ' ' + opts + ' ' + ' -f ' + out_dir + ' -s ' + decision_out_file
    print 'Running ',cmd
    os.system(cmd)
    
    score_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.rawscore.xml'
  
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1.ecf.xml'
    decision_out_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.rawscore.decision.xml'
    cmd = thresh_bin + ' ' + gt_file + ' ' + score_file + ' > ' + decision_out_file
    print 'Running ',cmd
    os.system(cmd)    
    
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.scoring.ecf.xml'
    rttm_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.mitllfa3.rttm'
    t_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-eval.kwlist2.xml'
    opts = '-o -b -O -B -c'
    out_dir = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.rawscore.decision'
    cmd = eval_bin + ' -e ' + gt_file + ' -r ' + rttm_file + ' -t ' + t_file + ' ' + opts + ' ' + ' -f ' + out_dir + ' -s ' + decision_out_file
    print 'Running ',cmd
    os.system(cmd)
    
    
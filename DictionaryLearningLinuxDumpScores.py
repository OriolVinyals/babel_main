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
    posting_file = './data/eval_part1.alignment.csv'
    perc_pos = 0.0
    babel_eval = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos,min_dur=min_dur)
    
    list_file = './data/20130307.eval.post.untightened.scp'
    feat_range = None
    babel_eval_post = BabelDataset.BabelDataset(list_file, feat_range, posting_file, perc_pos, keep_full_utt=True, posting_sampler=babel_eval.posting_sampler,min_dur=min_dur)
    
    '''Pipeline that just gets the score'''
    Xp_score = np.asmatrix(babel._features).T

    '''Constructing Dictionary of Features'''    
    Xtrain_dict = {'Score':Xp_score}
    Ytrain = babel.labels().astype(np.int)

    Xp_t_score = np.asmatrix(babel_eval._features).T
    Xtest_dict = {'Score':Xp_t_score}
    Ytest = babel_eval.labels().astype(np.int)

    lr_classifier = Classifier.Classifier(Xtrain_dict, Ytrain)
    '''Classifier stage'''
    feat_list=['Score']
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
    
    babel_eval.DumpScoresXML('./data/eval.scoreonly.xml',prob[:,1])
    babel_eval.DumpScoresXML('./data/eval.rawscore.xml',np.asarray(Xp_t_score).squeeze())
    
    score_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.scoreonly.xml'

    thresh_bin = '/u/swegmann/work/std/kws/score_norm/thresh.pl'
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1.ecf.xml'
    decision_out_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.scoreonly.decision.xml'
    cmd = thresh_bin + ' ' + gt_file + ' ' + score_file + ' > ' + decision_out_file
    print 'Running ',cmd
    os.system(cmd)
    
    eval_bin = 'perl5.14.2 /u/drspeech/projects/swordfish/ThirdParty/F4DE/bin/KWSEval'
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.scoring.ecf.xml'
    rttm_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.mitllfa3.rttm'
    t_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-eval.kwlist2.xml'
    opts = '-o -b -O -B -c'
    out_dir = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.scoreonly.decision'
    cmd = eval_bin + ' -e ' + gt_file + ' -r ' + rttm_file + ' -t ' + t_file + ' ' + opts + ' ' + ' -f ' + out_dir + ' -s ' + decision_out_file
    print 'Running ',cmd
    os.system(cmd)
    
    out_dir = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.rawscore.decision'
    cmd = eval_bin + ' -e ' + gt_file + ' -r ' + rttm_file + ' -t ' + t_file + ' ' + opts + ' ' + ' -f ' + out_dir + ' -s ' + '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.rawscore.xml'
    print 'Running ',cmd
    os.system(cmd)
    
    
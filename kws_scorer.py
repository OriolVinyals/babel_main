import string
import subprocess
import util

def get_score(score_file):
    #score_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.localutt.xml'
    
    thresh_bin = '/u/swegmann/work/std/kws/score_norm/thresh.pl'
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1.ecf.xml'
    
    decision_out_file = string.join(score_file.split('.')[0:-1],'.') + '.decision.xml'
    #decision_out_file = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.localutt.decision.xml'
    cmd = thresh_bin + ' ' + gt_file + ' ' + score_file + ' > ' + decision_out_file
    print 'Running ',cmd
    p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print out
    print err
    
    eval_bin = 'perl5.14.2 /u/drspeech/projects/swordfish/ThirdParty/F4DE/bin/KWSEval'
    gt_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.scoring.ecf.xml'
    rttm_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-evalpart1/babel104b-v0.4bY_conv-evalpart1.mitllfa3.rttm'
    t_file = '/u/drspeech/projects/swordfish/IndusDB/IndusDB.latest/babel104b-v0.4bY_conv-eval.kwlist2.xml'
    opts = '-o -b -O -B -c'
    out_dir = string.join(decision_out_file.split('.')[0:-1],'.')
    #out_dir = '/u/vinyals/projects/swordfish/src/python/babel_main/data/eval.localutt.decision'
    cmd = eval_bin + ' -e ' + gt_file + ' -r ' + rttm_file + ' -t ' + t_file + ' ' + opts + ' ' + ' -f ' + out_dir + ' -s ' + decision_out_file
    #print 'Running ',cmd
    p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()    
    #TODO return the ATWV!
    
    with open(out_dir + '.sum.txt') as f:
        last_line = util.tail(f,1)
        return float(last_line.split('|')[12])
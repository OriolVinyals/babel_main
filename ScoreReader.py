'''
Created on Jun 18, 2013

@author: vinyals
'''

import string
import numpy as np
import util
import os
import cPickle as pickle
import time
import diagnostics

def secondsToStr(t):
    rediv = lambda ll,b : list(divmod(ll[0],b)) + ll[1:]
    return "%d:%02d:%02d.%03d" % tuple(reduce(rediv,[[t*1000,],1000,60,60]))

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

class ScoreReader:
    def __init__(self,score_file,pickle_fname=None,list_times_utt_np=None):
        self.score_file = score_file
        self.map_utt_idx = {}
        self.pickle_fname = pickle_fname
        self.GetScoresXML(score_file)
        self.utt_feature = {}
        self.glob_feature = {}
        #self.num_utt = len(self.list_files)
        self.samp_period = 100
        self.list_times_utt_np = list_times_utt_np
            
    def GetScoresXML(self,fname):
        # We get every single entry so that we can pickle and load (since this is quite slow)
        # TODO: Pickle it!
        self.score_kw_utt_times_hash = AutoVivification()
        import xml.etree.cElementTree as ET
        tree = ET.parse(fname)
        root = tree.getroot()
        
        for i in range(len(root)):
            keyword = root[i].attrib['kwid']
            for j in range(len(root[i])):
                utterance = root[i][j].attrib['file']
                tbeg = root[i][j].attrib['tbeg']
                dur = root[i][j].attrib['dur']
                times = (round(float(tbeg),2),round(float(tbeg)+float(dur),2))
                score = root[i][j].attrib['score']
                #recursive dictionary
                self.score_kw_utt_times_hash[keyword][utterance][times] = float(score)
                self.map_utt_idx[utterance]=1
                #key = keyword + '_' + utterance + '_' + repr(times)
                #self.score_kw_utt_times_hash[key] = float(score)
        
    def GetKeywordData(self, utt_name, t_ini, t_end, kw=''):
        ret = self.score_kw_utt_times_hash[kw][utt_name][(t_ini,t_end)]
        if ret == {}:
            print 'Error couldnt find key!'
            exit(0)
        else:
            return ret
    
    def GetGlobFeature(self, utt_name, feat_type=['avg']):
        if self.glob_feature.has_key(utt_name):
            return self.glob_feature[utt_name]
        else:
            print 'Global Feature should have been precomputed!'
            exit(0)
    
    def GetUtteranceFeature(self, utt_name, times, feat_type=['avg']):
        utt_times = self.GetTimesUtterance(utt_name, times) #convert in utterance times to boundary utterance times
        utt_id_times = utt_name + '_' + '%07d' % (utt_times[0],) + '_' + '%07d' % (utt_times[1],)
        if self.utt_feature.has_key(utt_id_times):
            return self.utt_feature[utt_id_times]
        else:
            print 'Utterance Feature should have been precomputed!'
            exit(0)
    
    def GetTimesUtterance(self, utt_name, times):
        time_ind = (times[0]+times[1])/2*self.samp_period
        #utt_times = np.asarray(self.list_times_utt[utt_name])
        utt_times = self.list_times_utt_np[utt_name]
        #if np.any(utt_times==time_ind):
        #    print 'Warn: ',repr(utt_times)
        #    print 'Warn: ',repr(times)
        #    print 'Warn: ',utt_name
        #return np.squeeze(np.asarray(utt_times[np.nonzero(np.sum(time_ind<utt_times,axis=1)>0)[0][0]]))              
        return utt_times[np.nonzero(np.sum(time_ind<utt_times,axis=1)>0)[0][0]]        

if __name__ == '__main__':
    score_reader = ScoreReader('./data/word.kwlist.raw.xml')
    print score_reader.GetKeywordData('BABEL_BP_104_89382_20120207_192751_inLine', 310.87, 311.17, 'KW104-0055')
    print 'Should be 0.005'
    print 'Should produce error'
    print score_reader.GetKeywordData('muja!', 310.87, 311.17, 'KW104-0055')
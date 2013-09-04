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
    def __init__(self,score_file,pickle_fname=None,list_file_sph=None):
        self.score_file = score_file
        self.map_utt_idx = {}
        self.pickle_fname = pickle_fname
        self.ParseListScp(list_file_sph)
        self.GetScoresXML(score_file)
        self.utt_feature = {}
        self.glob_feature = {}
        #self.num_utt = len(self.list_files)
        self.samp_period = 100
        
    def ParseListScp(self, list_file):
        if list_file == None:
            return
        list_files = []
        self.list_times_utt = {}
        self.list_times_utt_np = {}
        with open(list_file) as f:
            for line in f:
                list_files.append(string.join(line.strip().split('_')[0:-2],'_') + '.sph')
                times=[]
                times.append(line.strip().split('_')[-2])
                times.append(line.strip().split('_')[-1].split('.')[0])
                utt_id = string.join(string.split(line.strip(),'/')[-1].split('_')[0:-2],'_')
                if self.list_times_utt.has_key(utt_id):
                    self.list_times_utt[utt_id].append((float(times[0]),float(times[1])))
                else:
                    self.list_times_utt[utt_id]=[]
                    self.list_times_utt[utt_id].append((float(times[0]),float(times[1])))
        list_files = set(list_files)
        for key in self.list_times_utt.keys():
            self.list_times_utt[key].sort(key=lambda x: x[0])
            self.list_times_utt_np[key] = np.asarray(self.list_times_utt[key])
        return [n for n in list_files]
            
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
                self.score_kw_utt_times_hash[utterance][times][keyword] = float(score)
                self.map_utt_idx[utterance]=1
                #key = keyword + '_' + utterance + '_' + repr(times)
                #self.score_kw_utt_times_hash[key] = float(score)
        
    def GetKeywordData(self, utt_name, t_ini, t_end, kw=''):
        vector_return = []
        ret = self.score_kw_utt_times_hash[utt_name][(t_ini,t_end)][kw]
        if ret == {}:
            print 'Error couldnt find key!'
            exit(0)
        else:
            vector_return.append(ret) # raw score ALWAYS first
            #vector_return.append(ret)
            return vector_return
            #return ret
    
    def GetGlobFeature(self, utt_name, feat_type=['avg']):
        if self.glob_feature.has_key(utt_name):
            return self.glob_feature[utt_name]
        else:
            #get all the scores from the glob file
            scores = [0.0]
            for times in self.score_kw_utt_times_hash[utt_name].values():
                for score in times.values():
                    scores.extend([score])
            vector_return = []
            for i in range(len(feat_type)):
                if feat_type[i] == 'avg':
                    vector_return.append(np.average(scores))
            self.glob_feature[utt_name] = vector_return
            return self.glob_feature[utt_name]
    
    def GetUtteranceFeature(self, utt_name, times, feat_type=['avg']):
        utt_times = self.GetTimesUtterance(utt_name, times) #convert in utterance times to boundary utterance times
        utt_id_times = utt_name + '_' + '%07d' % (utt_times[0],) + '_' + '%07d' % (utt_times[1],)
        if self.utt_feature.has_key(utt_id_times):
            return self.utt_feature[utt_id_times]
        else:
            scores = [0.0]
            for times in self.score_kw_utt_times_hash[utt_name]:
                if((times[0]*self.samp_period > utt_times[0]) and (times[1]*self.samp_period < utt_times[1])):
                    for score in self.score_kw_utt_times_hash[utt_name][times].values():
                        scores.extend([score])
            if len(scores)>1:
                del scores[0]
            vector_return = []
            for i in range(len(feat_type)):
                if feat_type[i] == 'avg':
                    vector_return.append(np.average(scores))
                if feat_type[i] == 'min':
                    vector_return.append(np.min(scores))
                if feat_type[i] == 'max':
                    vector_return.append(np.max(scores))
                if feat_type[i] == 'avg_log_odd':
                    aux = np.minimum(0.999,scores)
                    elem = aux / (1.0 - aux)
                    elem = np.log(elem)  
                    vector_return.append(np.average(elem))  
            self.utt_feature[utt_id_times] = vector_return
            return self.utt_feature[utt_id_times]
    
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
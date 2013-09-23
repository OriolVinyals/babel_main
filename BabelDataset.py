'''The Babel dataset 
'''

import cPickle as pickle
from iceberk import datasets, mpi
import numpy as np
import os
import itertools
import UtteranceReader
import LatticeReader
import SNRReader
import SrateReader
import PostingParser
import ScoreReader
import Sampler

class BabelDataset(datasets.ImageSet):
    """The Bable dataset
    """
    # some  Babel constants
    
    #def __init__(self, utt_reader,posting_sampler):
    def __init__(self, list_file, feat_range, posting_file, perc_pos, keep_full_utt=False, posting_sampler=None, min_dur=0.2, min_count=0.0, max_count=10000000.0, reader_type='utterance', 
                 pickle_fname=None, list_file_sph=None, kw_feat=None):
        '''TODO: Read pieces of utterance from the CSV file instead to save memory. It would be nice to index thse by utt_id (by now I do a map).'''
        super(BabelDataset, self).__init__()
        if list_file.find('eval') >= 0:
            self.is_eval = True
            self.T = 18600.705
        else:
            self.is_eval = False
            self.T = 36255.58
        self.beta = 999.9
        self.reader_type = reader_type
        if reader_type=='lattice':
            self.is_lattice = True
            utt_reader = LatticeReader.LatticeReader(list_file)
            utt_reader.ReadAllLatices()
        elif reader_type=='utterance':
            self.is_lattice = False
            utt_reader = UtteranceReader.UtteranceReader(list_file,pickle_fname=pickle_fname)
            utt_reader.ReadAllUtterances(feat_range)
        elif reader_type=='snr':
            self.is_lattice = False
            utt_reader = SNRReader.SNRReader(list_file,pickle_fname=pickle_fname)
            utt_reader.ReadAllSNR()
        elif reader_type=='srate':
            self.is_lattice = False
            utt_reader = SrateReader.SrateReader(list_file,pickle_fname=pickle_fname)
            utt_reader.ReadAllSrate()
        elif reader_type=='score':
            self.is_lattice = False
            utt_reader = ScoreReader.ScoreReader(list_file,list_file_sph=list_file_sph,pickle_fname=pickle_fname)
        else:
            print 'Reader not implemented!'
            exit(0)
        if posting_sampler == None:
            testParser = PostingParser.PostingParser(posting_file)
            self.posting_sampler = Sampler.Sampler(testParser)
            self.posting_sampler.GetPositive()
            self.posting_sampler.GetNegative()
            self.posting_sampler.SampleData(perc_pos)
        else:
            self.posting_sampler = posting_sampler
        self.min_dur = min_dur
        self._data_all = None
        self._dim = False
        self._channels = 1
        self.keep_full_utt = keep_full_utt
        if mpi.is_root():
            self._data = []
            self._label = []
            self._features = []
            self._utt_id = []
            self._times = []
            self._keyword = []
            skipped = 0
            for i in range(len(self.posting_sampler.negative_data)):
                if utt_reader.map_utt_idx.has_key(self.posting_sampler.negative_data[i]['file']):
                    if self.posting_sampler.negative_data[i]['sys_bt'] == '':
                        print 'We found a negative example that was not produced by the system!'
                        exit(0)
                    sys_bt = float(self.posting_sampler.negative_data[i]['sys_bt'])
                    sys_et = float(self.posting_sampler.negative_data[i]['sys_et'])
                    sys_sc = float(self.posting_sampler.negative_data[i]['sys_score'])
                    if(sys_et-sys_bt < self.min_dur):
                        skipped += 1
                        continue
                    self._data.append(utt_reader.GetKeywordData(self.posting_sampler.negative_data[i]['file'],
                                                              sys_bt, sys_et,kw=self.posting_sampler.negative_data[i]['termid']))
                    self._label.append(0)
                    self._features.append(sys_sc)
                    self._utt_id.append(self.posting_sampler.negative_data[i]['file'])
                    self._times.append((sys_bt,sys_et))
                    self._keyword.append(self.posting_sampler.negative_data[i]['termid'])
                else:
                    pass
            for i in range(len(self.posting_sampler.positive_data)):
                if utt_reader.map_utt_idx.has_key(self.posting_sampler.positive_data[i]['file']):
                    if self.posting_sampler.positive_data[i]['sys_bt'] == '':
                        sys_bt = 0
                        sys_et = None
                        sys_sc = -1.0
                        #print self.posting_sampler.positive_data[i]['alignment']
                        continue #Should just ignore these?
                    else:
                        sys_bt = float(self.posting_sampler.positive_data[i]['sys_bt'])
                        sys_et = float(self.posting_sampler.positive_data[i]['sys_et'])
                        sys_sc = float(self.posting_sampler.positive_data[i]['sys_score'])
                        if(sys_et-sys_bt < self.min_dur):
                            skipped += 1
                            continue
                    self._data.append(utt_reader.GetKeywordData(self.posting_sampler.positive_data[i]['file'],
                                                              sys_bt, sys_et,kw=self.posting_sampler.positive_data[i]['termid']))
                    self._label.append(1)
                    self._features.append(sys_sc)
                    self._utt_id.append(self.posting_sampler.positive_data[i]['file'])
                    self._times.append((sys_bt,sys_et))
                    self._keyword.append(self.posting_sampler.positive_data[i]['termid'])
                else:
                    pass
            
            print 'I skipped ',skipped,' entries'
            
            self._label = np.array(self._label)
        else:
            self._data = None
            self._label = None
            self._features = None
            self._utt_id = None
            self._times = None
            self._keyword = None
        #populate true kw freq
        self._map_kw_counts = {}
        for i in range(len(self.posting_sampler.positive_data)):
            if utt_reader.map_utt_idx.has_key(self.posting_sampler.positive_data[i]['file']):
                kw = self.posting_sampler.positive_data[i]['termid']
                if self._map_kw_counts.has_key(kw):
                    self._map_kw_counts[kw] += 1
                else:
                    self._map_kw_counts[kw] = 1
        #filter dataset depending on count
        if mpi.is_root():
            ind_keep = []
            kw_zero = 0
            for i in range(len(self._keyword)):
                kw = self._keyword[i]
                kw_count = 0
                if self._map_kw_counts.has_key(kw):
                    kw_count = self._map_kw_counts[kw]
                else:
                    kw_zero += 1
                if kw_count <= max_count and kw_count >= min_count:
                    ind_keep.append(i)
            
            self._data = [self._data[i] for i in ind_keep]
            self._label = [self._label[i] for i in ind_keep]
            self._features = [self._features[i] for i in ind_keep]
            self._utt_id = [self._utt_id[i] for i in ind_keep]
            self._times = [self._times[i] for i in ind_keep]
            self._keyword = [self._keyword[i] for i in ind_keep]

                    
        self._data = mpi.distribute_list(self._data)
        self._label = mpi.distribute(self._label)
        self._features = mpi.distribute_list(self._features)
        self._utt_id = mpi.distribute_list(self._utt_id)
        self._times = mpi.distribute_list(self._times)
        self._keyword = mpi.distribute_list(self._keyword)
        if self.keep_full_utt == True:
            self.utt_reader = utt_reader
        if kw_feat != None:
            try:
                kw_feat.has_key('length')
                self.CopyKeywordMaps(kw_feat)
            except:
                self.LoadMappingHescii('./data/hescii_babel104b-v0.4bY_conv-eval.kwlist2.xml')
                self.ComputeKeywordMaps()
                
        
    def ConvertFeatures(self,feat_range):
        '''Saves a copy for _data (all features), and strips out some features'''
        if self._data_all == None:
            self._data_all = self._data #make a copy the first time
        self._data = []
        for i in range(len(self._data_all)):
            self._data.append(self._data_all[i][:,feat_range])
            
    def GetLocalFeatures(self, feat_type=['entropy','entropy'],feat_range=None):
        '''Computes local features. Each mpi node computes its own'''
        self._local_features = []
        for i in range(len(self._data)):
            vector_return = []
            for j in range(len(feat_type)):
                if feat_type[j] == 'entropy':
                    aux = self._data[i]*np.log(self._data[i])
                    aux = np.sum(aux,1)
                    vector_return.append(np.average(aux))
                if feat_type[j] == 'duration':
                    vector_return.append(self._times[i][1]-self._times[i][0])
                if feat_type[j] == 'raw': #useful for lattices
                    if feat_range==None:
                        elem = self._data[i]
                    else:
                        elem = [self._data[i][j] for j in feat_range]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'raw_odd':
                    if feat_range==None:
                        aux = np.minimum(0.999,self._data[i])
                    else:
                        aux = np.minimum(0.999,[self._data[i][j] for j in feat_range])
                    elem = aux / (1.0 - aux) 
                    elem = elem.tolist()       
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'raw_log_odd':
                    #feat_range_t = [0,1,2,3,4,5,6,7,8]
                    feat_range_t = [0] #Only get first, which is original score
                    if feat_range_t==None:
                        aux = np.minimum(0.999,self._data[i])
                    else:
                        aux = np.minimum(0.999,[self._data[i][j] for j in feat_range_t])
                    elem = aux / (1.0 - aux)
                    elem = np.log(elem)
                    elem = elem.tolist()
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_length':
                    elem = self.map_keyword_feat['length'][self._keyword[i]]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_freq':
                    elem = self.map_keyword_feat['freq'][self._keyword[i]]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_freq_fine':
                    elem = self.map_keyword_feat['freq_fine'][self._keyword[i]]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_true_freq':
                    elem = self.map_keyword_feat['true_freq'][self._keyword[i]]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_true_freq_fine':
                    elem = self.map_keyword_feat['true_freq_fine'][self._keyword[i]]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_true_ratio':
                    elem = self.map_keyword_feat['true_ratio'][self._keyword[i]]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_id':
                    elem = self.map_keyword_feat['id'][self._keyword[i]]
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_threshold':
                    S = float(self.map_keyword_feat['n_est'][self._keyword[i]])
                    elem = (S)/(float(self.T)/float(self.beta) + S)
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_n_est_odd':
                    aux = float(self.map_keyword_feat['n_est'][self._keyword[i]]) / float(self.T)
                    aux = np.min((0.999,aux))
                    elem = float(self.beta) * aux / (1.0 - aux)
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_n_est':
                    aux = float(self.map_keyword_feat['n_est'][self._keyword[i]])
                    aux = aux + 0.1
                    elem = aux
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
                if feat_type[j] == 'kw_n_est_log_odd':
                    aux = float(self.map_keyword_feat['n_est'][self._keyword[i]]) / float(self.T)
                    aux = np.min((0.999,aux))
                    elem = float(self.beta) * aux / (1.0 - aux)
                    elem = np.log(elem)
                    if isinstance(elem,list):
                        vector_return.extend(elem)
                    else:
                        vector_return.append(elem)
            self._local_features.append(vector_return)
            
    def GetGlobalFeatures(self, feat_type=['entropy','entropy']):
        '''Computes global features. Each mpi node computes its own'''
        if self.keep_full_utt == False or self.is_lattice:
            print 'Error, we need to keep full utterance to compute global (per utterance) features! Or Lattice doesnt have global features!'
            exit(0)
        self._glob_features = []
        for i in range(len(self._utt_id)):
            self._glob_features.append(self.utt_reader.GetGlobFeature(self._utt_id[i], feat_type=feat_type))
            
    def GetUtteranceFeatures(self, feat_type=['entropy','entropy']):
        '''Computes utterance features. Each mpi node computes its own'''
        if self.keep_full_utt == False:
            print 'Error, we need to keep full utterance to compute utterance features!'
            exit(0)
        self._utt_features = []
        for i in range(len(self._utt_id)):
            self._utt_features.append(self.utt_reader.GetUtteranceFeature(self._utt_id[i], self._times[i], feat_type=feat_type))
            
    def DumpScoresXML(self,fname,scores=None):
        out_str = ''
        '''Header'''
        out_str += '<kwslist kwlist_filename="blah" language="vietnamese" system_id="blah">\n'
        self.KeywordSortedScores(scores)
        for keyword in self.keyword_scores.keys():
            out_str += '<detected_kwlist kwid="' + keyword + '" search_time="1" oov_count="0">\n'
            out_str += self.keyword_scores[keyword]
            out_str += '</detected_kwlist>\n'
        out_str += '</kwslist>'
        f = open(fname, 'w')
        f.write(out_str)
        f.close()
        return
    
    def KeywordSortedScores(self,scores=None):
        self.keyword_scores = {}
        for i in range(len(self._keyword)):
            kw_id=self._keyword[i]
            file=self._utt_id[i]
            times=self._times[i]
            if scores==None:
                score=self._features[i]
            else:
                score=scores[i]
            if score>=0.5:
                decision = 'YES'
            else:
                decision= 'NO'
            if self.keyword_scores.has_key(kw_id):
                self.keyword_scores[kw_id]+='<kw file="' + file + '" channel="1" tbeg="' + str(times[0]) + '" dur="' + str(times[1]-times[0]) + '" score="' + str(score) + '" decision="' + decision + '"/>\n'
            else:
                self.keyword_scores[kw_id]='<kw file="' + file + '" channel="1" tbeg="' + str(times[0]) + '" dur="' + str(times[1]-times[0]) + '" score="' + str(score) + '" decision="' + decision + '"/>\n'
    
    def GetATWV(self,scores,ths=None,compute_th=False):
        if compute_th:
            S = {}
            for i in range(len(self._keyword)):
                kw_id=self._keyword[i]
                if S.has_key(kw_id):
                    S[kw_id] += scores[i]
                else:
                    S[kw_id] = scores[i]
            ths = []
            for i in range(len(self._keyword)):
                ths.append((S[self._keyword[i]])/(float(self.T)/float(self.beta) + S[self._keyword[i]]))
        atwv = 0.0            
        for i in range(len(self._keyword)):
            score = scores[i]
            if ths == None:
                th = 0.5
            else:
                th = ths[i]
            if score>=th:
                kw_id=self._keyword[i]
                if self._map_kw_counts.has_key(kw_id):
                    if self._label[i] == 1:
                        atwv += 1.0/float(self._map_kw_counts[kw_id])
                    else:
                        atwv -= float(self.beta)/(float(self.T) - float(self._map_kw_counts[kw_id]))
        return atwv/len(self._map_kw_counts)
    
    def GetATWVsmooth(self,scores,ths=None,compute_th=False,method='sigmoid',factor=5.0,inds=None):
        if compute_th:
            S = {}
            for i in range(len(self._keyword)):
                kw_id=self._keyword[i]
                if S.has_key(kw_id):
                    S[kw_id] += scores[i]
                else:
                    S[kw_id] = scores[i]
            ths = []
            for i in range(len(self._keyword)):
                ths.append((S[self._keyword[i]])/(float(self.T)/float(self.beta) + S[self._keyword[i]]))
        try:
            weight = self._weight
        except:
            self._weight = np.zeros((len(self._keyword),))
            for i in range(len(self._keyword)):
                kw_id=self._keyword[i]
                if self._map_kw_counts.has_key(kw_id):
                    if self._label[i] == 1:
                        self._weight[i] = 1.0/float(self._map_kw_counts[kw_id])
                    else:
                        self._weight[i] = -float(self.beta)/(float(self.T) - float(self._map_kw_counts[kw_id]))
            weight = self._weight
        if ths==None:
            ths = 0.5
        if inds!=None:
            weight = weight[inds]
        
        loss,dloss = self.Loss01smooth(scores-ths,method=method,factor=factor)
        atwv = np.dot(loss, weight)/len(self._map_kw_counts)
        gatwv = dloss*weight/len(self._map_kw_counts)
        return atwv, gatwv
    
    @staticmethod
    def Loss01smooth(input,factor=10.0,method='sigmoid'):
        if method=='exact':
            out = input >= 0
            gout = input == 0
        elif method=='hinge':
            out = np.max(input + 1,0)
            gout = input + 1 >= 0
        elif method=='sigmoid':
            out = 1/(1+np.exp(-factor*(input)))
            gout = out*(1-out)*factor
        return out, gout
                        
    def LoadMappingHescii(self, fname):
        self.map_keyword = {}
        self.map_keyword_length = {}
        self.map_hescii = {}
        import xml.etree.cElementTree as ET
        tree = ET.parse(fname)
        root = tree.getroot()
        for i in range(len(root)):
            keyword = root[i].attrib['kwid']
            keyword_hescii = root[i][0].text
            self.map_keyword[keyword] = keyword_hescii
            self.map_keyword_length[keyword] = (len(keyword_hescii)-1)/2
            self.map_hescii[keyword_hescii] = keyword
            
    def ComputeKeywordMaps(self):
        self.map_keyword_feat = {}
        self.map_keyword_feat['length'] = self.map_keyword_length
        keyword_count = {}
        true_keyword_count = {}
        for i in range(len(self._keyword)):
            if keyword_count.has_key(self._keyword[i]):
                keyword_count[self._keyword[i]] += 1
            else:
                keyword_count[self._keyword[i]] = 1
            if self._label[i] == 1:
                if true_keyword_count.has_key(self._keyword[i]):
                    true_keyword_count[self._keyword[i]] += 1
                else:
                    true_keyword_count[self._keyword[i]] = 1
        for kw in self.map_keyword_length.keys():
            if not keyword_count.has_key(kw):
                keyword_count[kw] = 0
            if not true_keyword_count.has_key(kw):
                true_keyword_count[kw] = 0
        sorted_list = np.sort(keyword_count.values())
        th_25, th_50, th_75 = np.percentile(sorted_list, (25,50,75))
        self.map_keyword_feat['freq'] = {}
        for kw in self.map_keyword_length.keys():
            ret_vector = np.zeros((4))
            ret_vector[np.where(keyword_count[kw] > np.array((-1,th_25,th_50,th_75)))[0][-1]] = 1
            self.map_keyword_feat['freq'][kw] = ret_vector.tolist()
            
        th_vector = np.percentile(sorted_list, (10,20,30,40,50,60,70,80,90))
        th_vector.insert(0,-1)
        self.map_keyword_feat['freq_fine'] = {}
        for kw in self.map_keyword_length.keys():
            ret_vector = np.zeros((len(th_vector)))
            ret_vector[np.where(keyword_count[kw] > np.array(th_vector))[0][-1]] = 1
            self.map_keyword_feat['freq_fine'][kw] = ret_vector.tolist()
        
        sorted_list = np.sort(true_keyword_count.values())
        th_25, th_50, th_75 = np.percentile(sorted_list, (25,50,75))
        self.map_keyword_feat['true_freq'] = {}
        for kw in self.map_keyword_length.keys():
            ret_vector = np.zeros((4))
            ret_vector[np.where(true_keyword_count[kw] > np.array((-1,th_25,th_50,th_75)))[0][-1]] = 1
            self.map_keyword_feat['true_freq'][kw] = ret_vector.tolist()
            
        th_vector = np.percentile(sorted_list, (10,20,30,40,50,60,70,80,90))
        th_vector.insert(0,-1)
        self.map_keyword_feat['true_freq_fine'] = {}
        for kw in self.map_keyword_length.keys():
            ret_vector = np.zeros((len(th_vector)))
            ret_vector[np.where(true_keyword_count[kw] > np.array(th_vector))[0][-1]] = 1
            self.map_keyword_feat['true_freq_fine'][kw] = ret_vector.tolist()
            
        self.map_keyword_feat['true_ratio'] = {}
        for kw in self.map_keyword_length.keys():
            if keyword_count[kw] > 0:
                self.map_keyword_feat['true_ratio'][kw] = float(true_keyword_count[kw]) / float(keyword_count[kw])
            else:
                self.map_keyword_feat['true_ratio'][kw] = 0.0
            
        self.map_keyword_feat['id'] = {}
        oov_index = 0
        oov_th = 10
        num_keywords = len(np.nonzero(np.asarray(keyword_count.values())>oov_th)[0])
        aux = np.eye(num_keywords+1)
        index = 1
        for kw in keyword_count:
            if keyword_count[kw] <= oov_th:
                self.map_keyword_feat['id'][kw] = aux[oov_index].tolist()
            else:
                self.map_keyword_feat['id'][kw] = aux[index].tolist()
                index += 1
                
        if self.reader_type == 'score':
            self.map_keyword_feat['n_est'] = {}
            for i in range(len(self._data)):
                if self.map_keyword_feat['n_est'].has_key(self._keyword[i]):
                    self.map_keyword_feat['n_est'][self._keyword[i]] += self._data[i][0]
                else:
                    self.map_keyword_feat['n_est'][self._keyword[i]] = self._data[i][0]
        self.map_keyword_feat['n_true'] = true_keyword_count
        print 'Set of kyeword features computed'

        
    def CopyKeywordMaps(self, map_keyword_feat):
        print 'Loading previously set keyword features'
        self.map_keyword_feat = map_keyword_feat
        if self.reader_type == 'score': #This feature is dataset specific
            self.map_keyword_feat['n_est'] = {}
            for i in range(len(self._data)):
                if self.map_keyword_feat['n_est'].has_key(self._keyword[i]):
                    self.map_keyword_feat['n_est'][self._keyword[i]] += self._data[i][0]
                else:
                    self.map_keyword_feat['n_est'][self._keyword[i]] = self._data[i][0]
            
if __name__ == '__main__':
    feat_range = [0,1,2,5,6,7,69,74]
    posting_file = './data/word.kwlist.alignment.csv'
    perc_pos = 0.0
    list_file = './data/audio.list'
    babel_srate = BabelDataset(list_file, None, posting_file, perc_pos, keep_full_utt=True, reader_type='srate',pickle_fname='./pickles/full.srate.pickle')
    babel_srate.GetUtteranceFeatures('srate')
    babel_snr = BabelDataset(list_file, None, posting_file, perc_pos, reader_type='snr')
    list_file = './data/lat.debug.list'
    babel_lat = BabelDataset(list_file, None, posting_file, perc_pos, reader_type='lattice')
    list_file = './data/list_files.scp'
    babel = BabelDataset(list_file, None, posting_file, perc_pos, reader_type='utterance')
    #babel.GetLocalFeatures(feat_type=['score'],fname_xml='./data/word.kwlist.raw.xml')
    print babel._data[0].shape
    babel.ConvertFeatures([0,1,3])
    print babel._data[0].shape
    babel.ConvertFeatures([0,1,74])
    print babel._data[0].shape
    babel.ConvertFeatures(range(30))
    print babel._data[0].shape
    print babel._utt_id[0]
    babel.DumpScoresXML('./data/unittest.xml')

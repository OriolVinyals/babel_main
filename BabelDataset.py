'''The Babel dataset 
'''

import cPickle as pickle
from iceberk import datasets, mpi
import numpy as np
import os
import UtteranceReader
import PostingParser
import Sampler

class BabelDataset(datasets.ImageSet):
    """The Bable dataset
    """
    # some  Babel constants
    
    #def __init__(self, utt_reader,posting_sampler):
    def __init__(self, list_file, feat_range, posting_file, perc_pos, keep_full_utt=False, posting_sampler=None, min_dur=0.2):
        '''TODO: Read pieces of utterance from the CSV file instead to save memory. It would be nice to index thse by utt_id (by now I do a map).'''
        super(BabelDataset, self).__init__()
        utt_reader = UtteranceReader.UtteranceReader(list_file)
        utt_reader.ReadAllUtterances(feat_range)
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
        self._kw_utt_times_hash = {}
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
                    self._data.append(utt_reader.GetUtterance(self.posting_sampler.negative_data[i]['file'],
                                                              sys_bt, sys_et))
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
                    self._data.append(utt_reader.GetUtterance(self.posting_sampler.positive_data[i]['file'],
                                                              sys_bt, sys_et))
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
        self._data = mpi.distribute_list(self._data)
        self._label = mpi.distribute(self._label)
        self._features = mpi.distribute_list(self._features)
        self._utt_id = mpi.distribute_list(self._utt_id)
        self._times = mpi.distribute_list(self._times)
        self._keyword = mpi.distribute_list(self._keyword)
        if self.keep_full_utt == True:
            self.utt_reader = utt_reader
        
    def ConvertFeatures(self,feat_range):
        '''Saves a copy for _data (all features), and strips out some features'''
        if self._data_all == None:
            self._data_all = self._data #make a copy the first time
        self._data = []
        for i in range(len(self._data_all)):
            self._data.append(self._data_all[i][:,feat_range])
            
    def GetLocalFeatures(self, feat_type=['entropy','entropy'],fname_xml=None):
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
                    vector_return.append(self._data[i].shape[0]/float(100))
                if feat_type[j] == 'score':
                    self.GetScoresXML(fname_xml)
                    key = self._keyword[i] + '_' + self._utt_id[i] + '_' + repr(self._times[i])
                    vector_return.append(self._kw_utt_times_hash[key])
                    pass
            self._local_features.append(vector_return)
            
    def GetGlobalFeatures(self, feat_type=['entropy','entropy']):
        '''Computes global features. Each mpi node computes its own'''
        if self.keep_full_utt == False:
            print 'Error, we need to keep full utterance to compute global (per utterance) features!'
            exit(0)
        self._glob_features = []
        for i in range(len(self._utt_id)):
            self._glob_features.append(self.utt_reader.GetGlobFeature(self._utt_id[i], feat_type=feat_type))
            
    def GetUtteranceFeatures(self, feat_type=['entropy','entropy']):
        '''Computes global features. Each mpi node computes its own'''
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

    def GetScoresXML(self,fname):
        if len(self._kw_utt_times_hash) > 0:
            return
        from xml.dom import minidom
        xmldoc = minidom.parse(fname)
        itemlist = xmldoc.getElementsByTagName('detected_kwlist')
        
        for i in range(len(itemlist)):
            keyword = itemlist[i].attributes['kwid'].value
            for j in range(len(itemlist[i].childNodes)):
                if itemlist[i].childNodes[j].nodeType == itemlist[i].childNodes[j].ELEMENT_NODE:
                    utterance = itemlist[i].childNodes[j].attributes['file'].value
                    tbeg = itemlist[i].childNodes[j].attributes['tbeg'].value
                    dur = itemlist[i].childNodes[j].attributes['dur'].value
                    times = (round(float(tbeg),2),round(float(tbeg)+float(dur),2))
                    score = itemlist[i].childNodes[j].attributes['score'].value
                    key = keyword + '_' + utterance + '_' + repr(times) 
                    self._kw_utt_times_hash[key] = float(score)
        return

            
if __name__ == '__main__':
    list_file = './data/list_files.scp'
    feat_range = [0,1,2,5,6,7,69,74]
    posting_file = './data/word.kwlist.alignment.csv'
    perc_pos = 0.2
    babel = BabelDataset(list_file, None, posting_file, perc_pos)
    babel.GetLocalFeatures(feat_type=['score'],fname_xml='./data/word.kwlist.raw.xml')
    print babel._data[0].shape
    babel.ConvertFeatures([0,1,3])
    print babel._data[0].shape
    babel.ConvertFeatures([0,1,74])
    print babel._data[0].shape
    babel.ConvertFeatures(range(30))
    print babel._data[0].shape
    print babel._utt_id[0]
    babel.DumpScoresXML('./data/unittest.xml')

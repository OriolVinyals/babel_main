'''
Created on Jun 18, 2013

@author: vinyals
'''

from external import htkmfc
import string
import numpy as np
import cPickle as pickle
import os

class UtteranceReader:
    def __init__(self,list_file,pickle_fname=None):
        self.list_file = list_file
        self.list_files = self.ParseListScp(list_file)
        self.utt_data = []
        self.glob_feature = {}
        self.utt_feature = {}
        file_htk = htkmfc.HTKFeat_read(self.list_files[0])
        file_htk.readheader()
        self.feat_size = file_htk.veclen
        self.num_utt = len(self.list_files)
        self.samp_period = file_htk.sampPeriod/(1e9/100)
        self.map_utt_idx = {}
        self.pickle_fname = pickle_fname
        if pickle_fname != None:
            abs_path = os.path.abspath(pickle_fname)
            try:
                os.makedirs(os.path.dirname(abs_path))
            except:
                print 'Path exists or cant create'
         
    def ReadAllUtterances(self, feat_range=None):
        try:
            with open(self.pickle_fname,'rb') as fp:
                print 'Reading data from file',self.pickle_fname
                self.utt_data=pickle.load(fp)
                self.feat_range=pickle.load(fp)
                self.feat_size=pickle.load(fp)
                self.map_utt_idx=pickle.load(fp)
        except:
            if feat_range == None:
                feat_range = range(self.feat_size)
            self.feat_range = feat_range
            self.feat_size = len(feat_range)
            
            for i in range(len(self.list_files)):
                utt_id = string.split(self.list_files[i],'/')[-1]
                file_htk = htkmfc.HTKFeat_read(self.list_files[i])
                np_data = file_htk.getall()[:,self.feat_range]
                self.utt_data.append(np_data)
                self.map_utt_idx[utt_id] = i
            try:
                with open(self.pickle_fname,'wb') as fp:
                        pickle.dump(self.utt_data,fp)
                        pickle.dump(self.feat_range,fp)
                        pickle.dump(self.feat_size,fp)
                        pickle.dump(self.map_utt_idx,fp)
            except:
                print 'Could not open file for saving',self.pickle_fname
            
    def GetKeywordData(self, utt_name, t_ini=0, t_end=None, kw=''):
        if self.utt_data == []:
            print 'We need to read utterances first'
            return
        index = self.map_utt_idx[utt_name]
        t_ini = t_ini/self.samp_period
        if t_end==None:
            t_end=self.utt_data[index].shape[0]
        else:
            t_end=t_end/self.samp_period
        return self.utt_data[index][t_ini:t_end,:]
    
    def ParseListScp(self, list_file):
        list_files = []
        self.list_times_utt = {}
        self.list_times_utt_np = {}
        with open(list_file) as f:
            for line in f:
                list_files.append(line.strip().split('=')[1].strip().split('[')[0])
                times = line.strip().split('=')[1].strip().split('[')[1].strip().split(']')[0].split(',')
                utt_id = string.split(list_files[-1],'/')[-1]
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
    
    def GetGlobFeature(self, utt_name, feat_type=['entropy']):
        if self.utt_data == []:
            print 'We need to read utterances first'
            return
        if self.glob_feature.has_key(utt_name):
            return self.glob_feature[utt_name]
        index = self.map_utt_idx[utt_name]
        utt = self.utt_data[index]
        vector_return = []
        for i in range(len(feat_type)):
            if feat_type[i] == 'entropy':
                aux = utt*np.log(utt)
                aux = np.sum(aux,1)
                vector_return.append(np.average(aux))
        self.glob_feature[utt_name] = vector_return
        return self.glob_feature[utt_name]
    
    def GetUtteranceFeature(self, utt_name, times, feat_type=['entropy']):
        if self.utt_data == []:
            print 'We need to read utterances first'
            return
        utt_times = self.GetTimesUtterance(utt_name,times)
        #print utt_times
        #print utt_name
        #print times
        key = utt_name,np.array2string(utt_times)
        if self.utt_feature.has_key(key):
            return self.utt_feature[key]
        index = self.map_utt_idx[utt_name]
        utt = self.utt_data[index]
        #if utt_times[1]>utt.shape[0]:
        #    utt_times[1]=utt.shape[0]
        #    print utt_times[1],' vs ',utt.shape[0]
        utt = utt[utt_times[0]:utt_times[1],:]
        vector_return = []
        for i in range(len(feat_type)):
            if feat_type[i] == 'entropy':
                aux = utt*np.log(utt)
                aux = np.sum(aux,1)
                vector_return.append(np.average(aux))
        self.utt_feature[key] = vector_return
        return self.utt_feature[key]
    
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
    list_files = './data/list_files.scp'
    feat_range = [0,1,2,5,6,7,69,74]
    utt_reader = UtteranceReader(list_files)
    print utt_reader.feat_size
    print utt_reader.num_utt
    print utt_reader.samp_period
    utt_reader.ReadAllUtterances(feat_range)
    print utt_reader.utt_data
    utt_reader.ReadAllUtterances()
    print utt_reader.utt_data        
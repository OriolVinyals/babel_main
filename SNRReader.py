'''
Created on Jun 18, 2013

@author: vinyals
'''

import string
import numpy as np
import os

class SNRReader:
    def __init__(self,list_file):
        self.list_file = list_file
        self.list_files = self.ParseListScp(list_file)
        self.lat_data = []
        self.utt_data = []
        self.utt_feature = {}
        self.num_utt = len(self.list_files)
        self.samp_period = 100
        self.map_utt_idx = {}
        self.map_utt_times_idx = {}
         
    def ReadAllSNR(self):        
        #VERY time expensive. Computes Utterance and Global features (but no local features unlike Lat/UTTReader)
        import subprocess
        for i in range(len(self.list_files)):
            utt_id = string.split(self.list_files[i],'/')[-1].split('.')[0]
            for times in self.list_times_utt[utt_id]:
                t_beg = times[0]/self.samp_period
                t_end = times[1]/self.samp_period
                cmd = '/u/vinyals/projects/swordfish/src/snreval/run_snreval_prj.sh ' + self.list_files[i] + ' '
                cmd += '-start ' + repr(t_beg) + ' -end ' + repr(t_end) + ' -disp 0'
                #os.system(cmd)
                p = subprocess.Popen(cmd.split(' '), 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = p.communicate()
                print out, err
            #ENDTODO
            self.lat_data.append(np_data)
            self.map_utt_idx[utt_id] = i
            self.map_utt_times_idx[utt_id_times] = i
    
    def GetUtterance(self, utt_name, t_ini, t_end):
        # No per frame / local feature for SNR!
        return 0
    
    def ParseListScp(self, list_file):
        list_files = []
        self.list_times_utt = {}
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
        return [n for n in list_files]
    
    def GetGlobFeature(self, utt_name, feat_type='entropy'):
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
    
    def GetUtteranceFeature(self, utt_name, times, feat_type='somefeature'):
        utt_times = self.GetTimesUtterance(utt_name, times) #convert in utterance times to boundary utterance times
        utt_id_times = utt_name + '_' + '%07d' % (utt_times[0],) + '_' + '%07d' % (utt_times[1],)
        if self.utt_feature.has_key(utt_id_times):
            return self.utt_feature[utt_id_times]
        index = self.map_utt_times_idx[utt_id_times]
        lattice_data = self.lat_data[index]
        vector_return = []
        for i in range(len(feat_type)):
            if feat_type[i] == 'somefeature':
                pass
                #do something with lattice_data
        self.utt_feature[utt_id_times] = vector_return
        return self.utt_feature[utt_id_times]

    
    def GetTimesUtterance(self, utt_name, times):
        time_ind = (times[0]+times[1])/(2*self.samp_period)
        utt_times = np.asarray(self.list_times_utt[utt_name])
        #if np.any(utt_times==time_ind):
        #    print 'Warn: ',repr(utt_times)
        #    print 'Warn: ',repr(times)
        #    print 'Warn: ',utt_name
        return np.squeeze(np.asarray(utt_times[np.nonzero(np.sum(time_ind<utt_times,axis=1)>0)[0][0]]))              

if __name__ == '__main__':
    list_files = './data/audio.list'
    snr_reader = SNRReader(list_files)  
    snr_reader.ReadAllSNR()
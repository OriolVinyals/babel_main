'''
Created on Jun 18, 2013

@author: vinyals
'''

from external import lattice
import string
import numpy as np

class LatticeReader:
    def __init__(self,list_file):
        self.list_file = list_file
        self.list_files = self.ParseListScp(list_file)
        self.lat_data = []
        self.utt_data = []
        self.utt_feature = {}
        self.num_utt = len(self.list_files)
        self.samp_period = 100
        self.map_utt_idx = {}
         
    def ReadAllLatices(self):        
        for i in range(len(self.list_files)):
            utt_id_times = string.split(self.list_files[i],'/')[-1].split('.')[0]
            #TODO
            #np_data = lattice.Dag(htk_file=self.list_files[i])
            np_data = 0
            #ENDTODO
            self.lat_data.append(np_data)
            self.map_utt_idx[utt_id_times] = i
    
    def GetUtterance(self, utt_name, t_ini, t_end):
        # times in seconds
        times = self.GetTimesUtterance(utt_name, (t_ini,t_end))
        utt_id_times = utt_name + '_' + '%07d' % (times[0],) + '_' + '%07d' % (times[1],)
        index = self.map_utt_idx[utt_id_times]
        lattice_data = self.lat_data[index]
        rel_t_ini = t_ini - times[0]/self.samp_period
        rel_t_end = t_end - times[0]/self.samp_period

        #compute something with lat_data[index] and the times and return it
        return 0
    
    def ParseListScp(self, list_file):
        list_files = []
        self.list_times_utt = {}
        with open(list_file) as f:
            for line in f:
                list_files.append(line.strip())
                times=[]
                times.append(line.strip().split('_')[-2])
                times.append(line.strip().split('_')[-1].split('.')[0])
                utt_id = string.join(string.split(list_files[-1],'/')[-1].split('_')[0:-2],'_')
                if self.list_times_utt.has_key(utt_id):
                    self.list_times_utt[utt_id].append((float(times[0]),float(times[1])))
                else:
                    self.list_times_utt[utt_id]=[]
                    self.list_times_utt[utt_id].append((float(times[0]),float(times[1])))
        list_files = set(list_files)
        for key in self.list_times_utt.keys():
            self.list_times_utt[key].sort(key=lambda x: x[0])
        return [n for n in list_files]
    
    def GetUtteranceFeature(self, utt_name, times, feat_type='somefeature'):
        utt_times = self.GetTimesUtterance(utt_name, times) #convert in utterance times to boundary utterance times
        utt_id_times = utt_name + '_' + '%07d' % (utt_times[0],) + '_' + '%07d' % (utt_times[1],)
        if self.utt_feature.has_key(utt_id_times):
            return self.utt_feature[utt_id_times]
        index = self.map_utt_idx[utt_id_times]
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
    list_files = './data/lat.debug.list'
    lat_reader = LatticeReader(list_files)  
    lat_reader.ReadAllLatices()
    lat_reader.GetUtterance('BABEL_BP_104_85455_20120310_210107_outLine', 573.4, 573.84)
    # BABEL_BP_104_04221_20120310_194031_inLine_0000133_0000572
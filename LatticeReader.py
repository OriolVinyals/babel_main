'''
Created on Jun 18, 2013

@author: vinyals
'''

from external import lattice
import string

class LatticeReader:
    def __init__(self,list_file):
        self.list_file = list_file
        self.list_files = self.ParseListScp(list_file)
        self.lat_data = []
        self.num_utt = len(self.list_files)
        self.map_utt_idx = {}
        self.samp_period = 100
         
    def ReadAllLatices(self):        
        for i in range(len(self.list_files)):
            utt_id = string.split(self.list_files[i],'/')[-1].split('.')[0]
            np_data = lattice.Dag(htk_file=self.list_files[i])
            self.lat_data.append(np_data)
            self.map_utt_idx[utt_id] = i
            
    def GetLattice(self, utt_name):
        if self.utt_data == []:
            print 'We need to read utterances first'
            return
        index = self.map_utt_idx[utt_name]
        return self.lat_data[index]
    
    def ParseListScp(self, list_file):
        list_files = []
        self.list_times_utt = {}
        with open(list_file) as f:
            for line in f:
                list_files.append(line)
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
    
    def GetTimesUtterance(self, utt_name, times):
        time_ind = (times[0]+times[1])/(2*self.samp_period)
        utt_times = np.asarray(self.list_times_utt[utt_name])
        #if np.any(utt_times==time_ind):
        #    print 'Warn: ',repr(utt_times)
        #    print 'Warn: ',repr(times)
        #    print 'Warn: ',utt_name
        return np.squeeze(np.asarray(utt_times[np.nonzero(np.sum(time_ind<utt_times,axis=1)>0)[0][0]]))
                

if __name__ == '__main__':
    list_files = './data/lat.list'
    lat_reader = LatticeReader(list_files)  
    lat_reader.ReadAllLatices()
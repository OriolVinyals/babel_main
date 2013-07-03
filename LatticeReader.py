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
        with open(list_file) as f:
            for line in f:
                list_files.append(line)
        list_files = set(list_files)
        return [n for n in list_files]
                

if __name__ == '__main__':
    list_files = './data/lat.debug.list'
    lat_reader = LatticeReader(list_files)  
    lat_reader.ReadAllLatices()
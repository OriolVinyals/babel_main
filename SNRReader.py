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

class SNRReader:
    def __init__(self,list_file,pickle_fname='./pickles/test.snr.pickle'):
        self.list_file = list_file
        self.list_files = self.ParseListScp(list_file)
        self.utt_feature = {}
        self.glob_feature = {}
        self.num_utt = len(self.list_files)
        self.samp_period = 100
        self.map_utt_idx = {}
        self.pickle_fname = pickle_fname
        abs_path = os.path.abspath(pickle_fname)
        try:
            os.makedirs(os.path.dirname(abs_path))
        except:
            print 'Path exists or cant create'
         
    def ReadAllSNR(self):        
        #VERY time expensive. Computes Utterance and Global features (but no local features unlike Lat/UTTReader)
        try:
            with open(self.pickle_fname,'rb') as fp:
                self.utt_feature=pickle.load(fp)
                self.glob_feature=pickle.load(fp)
                self.map_utt_idx=pickle.load(fp)
        except:
            num_utt = len(self.list_files)
            for i in range(len(self.list_files)):
                utt_id = string.split(self.list_files[i],'/')[-1].split('.')[0]
                num_utt += len(self.list_times_utt[utt_id])
            avg_iter = 0
            curr_utt = 0
            curr_dir = os.getcwd()
            for i in range(len(self.list_files)):
                utt_id = string.split(self.list_files[i],'/')[-1].split('.')[0]
                audio_chunk = ''
                for times in self.list_times_utt[utt_id]:
                    t1 = time.time()
                    t_beg = times[0]/self.samp_period
                    t_end = times[1]/self.samp_period
                    utt_id_times = utt_id + '_' + '%07d' % (times[0],) + '_' + '%07d' % (times[1],)
                    self.utt_feature[utt_id_times] = util.cmdSNR(self.list_files[i], t_beg, t_end)
                    ellapsed = time.time() - t1
                    avg_iter = avg_iter + (ellapsed-avg_iter)/(curr_utt+1)
                    curr_utt += 1
                    audio_chunk += self.list_files[i] + ' ' + repr(t_beg) + ' ' + repr(t_end) + '\n'
                    if curr_utt%10 == 0:
                        print 'Iteration ' + repr(curr_utt) + ' out of ' + repr(num_utt)
                        print 'Time per iteration ' + '%.2f' % (avg_iter)
                        print 'ETA ' + secondsToStr(avg_iter*(num_utt-curr_utt))
                t1 = time.time()
                util.cmdChunk('./temp.snr.sph', audio_chunk)
                self.glob_feature[utt_id] = util.cmdSNR(curr_dir+'/temp.snr.sph')
                ellapsed = time.time() - t1
                avg_iter = avg_iter + (ellapsed-avg_iter)/(curr_utt+1)
                curr_utt += 1
                self.map_utt_idx[utt_id] = i
                print 'Iteration ' + repr(curr_utt) + ' out of ' + repr(num_utt)
                print 'Time per iteration ' + '%.2f' % (avg_iter)
                print 'ETA ' + secondsToStr(avg_iter*(num_utt-curr_utt))

            with open(self.pickle_fname,'wb') as fp:
                    pickle.dump(self.utt_feature,fp)
                    pickle.dump(self.glob_feature,fp)
                    pickle.dump(self.map_utt_idx,fp)
        
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
    
    def GetGlobFeature(self, utt_name, feat_type='snr'):
        if self.glob_feature.has_key(utt_name):
            return self.glob_feature[utt_name]
        else:
            print 'Global Feature should have been precomputed!'
            exit(0)
    
    def GetUtteranceFeature(self, utt_name, times, feat_type='snr'):
        utt_times = self.GetTimesUtterance(utt_name, times) #convert in utterance times to boundary utterance times
        utt_id_times = utt_name + '_' + '%07d' % (utt_times[0],) + '_' + '%07d' % (utt_times[1],)
        if self.utt_feature.has_key(utt_id_times):
            return self.utt_feature[utt_id_times]
        else:
            print 'Utterance Feature should have been precomputed!'
            exit(0)

    
    def GetTimesUtterance(self, utt_name, times):
        time_ind = (times[0]+times[1])/2*self.samp_period
        utt_times = np.asarray(self.list_times_utt[utt_name])
        #if np.any(utt_times==time_ind):
        #    print 'Warn: ',repr(utt_times)
        #    print 'Warn: ',repr(times)
        #    print 'Warn: ',utt_name
        return np.squeeze(np.asarray(utt_times[np.nonzero(np.sum(time_ind<utt_times,axis=1)>0)[0][0]]))              

if __name__ == '__main__':
    list_files = './data/audio.list'
    snr_reader = SNRReader(list_files,pickle_fname='./pickles/full.snr.pickle')  
    snr_reader.ReadAllSNR()
    diagnostics.print_histogram(snr_reader.glob_feature,'./data/plot_snr_glob.png')
    diagnostics.print_histogram(snr_reader.utt_feature,'./data/plot_snr_utt.png')
    print 'Utterance Feature ' + repr(snr_reader.GetUtteranceFeature('BABEL_BP_104_35756_20120311_223543_inLine',(294,295)))
    print 'Global Feature ' + repr(snr_reader.GetGlobFeature('BABEL_BP_104_35756_20120311_223543_inLine'))
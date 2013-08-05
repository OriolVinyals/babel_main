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

class SrateReader:
    def __init__(self,list_file,pickle_fname='./pickles/test.srate.pickle'):
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
         
    def ReadAllSrate(self):        
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
                    self.utt_feature[utt_id_times] = util.cmdmrate(self.list_files[i], t_beg, t_end)
                    ellapsed = time.time() - t1
                    avg_iter = avg_iter + (ellapsed-avg_iter)/(curr_utt+1)
                    curr_utt += 1
                    audio_chunk += self.list_files[i] + ' ' + repr(t_beg) + ' ' + repr(t_end) + '\n'
                    if curr_utt%100 == 0:
                        print 'Iteration ' + repr(curr_utt) + ' out of ' + repr(num_utt)
                        print 'Time per iteration ' + '%.2f' % (avg_iter)
                        print 'ETA ' + secondsToStr(avg_iter*(num_utt-curr_utt))
                t1 = time.time()
                util.cmdChunk('./temp.srate.sph', audio_chunk)
                self.glob_feature[utt_id] = util.cmdmrate(curr_dir+'/temp.srate.sph')
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
    
    def GetGlobFeature(self, utt_name, feat_type='srate'):
        if self.glob_feature.has_key(utt_name):
            return self.glob_feature[utt_name]
        else:
            print 'Global Feature should have been precomputed!'
            exit(0)
    
    def GetUtteranceFeature(self, utt_name, times, feat_type='srate'):
        utt_times = self.GetTimesUtterance(utt_name, times) #convert in utterance times to boundary utterance times
        utt_id_times = utt_name + '_' + '%07d' % (utt_times[0],) + '_' + '%07d' % (utt_times[1],)
        if self.utt_feature.has_key(utt_id_times):
            #DEBUG
            if(utt_times[0]>times[0]*100):
                if(utt_times[0]-times[0]*100 > 40):
                    print 'Muja'
                print utt_times[0]-times[0]*100
            if(utt_times[1]<times[1]*100):
                if(times[1]*100-utt_times[1] > 40):
                    print 'Muja2'
                print times[1]*100-utt_times[1]
            #DEBUG
            return self.utt_feature[utt_id_times]
        else:
            print 'Utterance Feature should have been precomputed!'
            exit(0)
            
    def DumpAudioDiagnostics(self,dir_name='./data/',top_k=10,bot_k=10):
        #utterance level diag
        import heapq
        utt_largest = heapq.nlargest(top_k, self.utt_feature, key=self.utt_feature.get)
        i=0
        for utt in utt_largest:
            utt_id = string.join(utt.split('_')[0:-2],'_')
            t_beg = float(utt.split('_')[-2])/self.samp_period
            t_end = float(utt.split('_')[-1])/self.samp_period
            file_id = self.list_files[self.map_utt_idx[utt_id]]
            out_file = './data/' + repr(i) + 'large_srate_' + os.path.basename(file_id).split('.')[0] + '.wav'
            util.cmdconvert(file_id, out_file, t_beg, t_end)
            i+=1
        utt_smallest = heapq.nsmallest(bot_k, self.utt_feature, key=self.utt_feature.get)
        i=0
        for utt in utt_smallest:
            utt_id = string.join(utt.split('_')[0:-2],'_')
            t_beg = float(utt.split('_')[-2])/self.samp_period
            t_end = float(utt.split('_')[-1])/self.samp_period
            file_id = self.list_files[self.map_utt_idx[utt_id]]
            out_file = './data/' + repr(i) + 'small_srate_' + os.path.basename(file_id).split('.')[0] + '.wav'
            util.cmdconvert(file_id, out_file, t_beg, t_end)
            i+=1
        #glob level diag
        glob_largest = heapq.nlargest(top_k, self.glob_feature, key=self.glob_feature.get)
        for utt_id in glob_largest:
            file_id = self.list_files[self.map_utt_idx[utt_id]]
            out_file = './data/glob_large_srate_' + os.path.basename(file_id).split('.')[0] + '.wav'
            util.cmdconvert(file_id, out_file)
        glob_smallest = heapq.nsmallest(top_k, self.glob_feature, key=self.glob_feature.get)
        for utt_id in glob_smallest:
            file_id = self.list_files[self.map_utt_idx[utt_id]]
            out_file = './data/glob_small_srate_' + os.path.basename(file_id).split('.')[0] + '.wav'
            util.cmdconvert(file_id, out_file)

    
    def GetTimesUtterance(self, utt_name, times):
        time_ind = (times[0]+times[1])/2*self.samp_period
        utt_times = np.asarray(self.list_times_utt[utt_name])
        #if np.any(utt_times==time_ind):
        #    print 'Warn: ',repr(utt_times)
        #    print 'Warn: ',repr(times)
        #    print 'Warn: ',utt_name
        return np.squeeze(np.asarray(utt_times[np.nonzero(np.sum(time_ind<utt_times,axis=1)>0)[0][0]]))              

if __name__ == '__main__':
    list_files = './data/audio.eval.list'
    srate_reader = SrateReader(list_files,pickle_fname='./pickles/full.eval.srate.pickle')  
    srate_reader.ReadAllSrate()
    srate_reader.DumpAudioDiagnostics()
    diagnostics.print_histogram(srate_reader.glob_feature,'./data/plot_srate_glob.png')
    diagnostics.print_histogram(srate_reader.utt_feature,'./data/plot_srate_utt.png')
    print 'Utterance Feature ' + repr(srate_reader.GetUtteranceFeature('BABEL_BP_104_35756_20120311_223543_inLine',(294,295)))
    print 'Global Feature ' + repr(srate_reader.GetGlobFeature('BABEL_BP_104_35756_20120311_223543_inLine'))
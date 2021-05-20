import os
import librosa
import config
import csv
import glob
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from intervaltree import IntervalTree,Interval
from utils import reduce_bg_noise,spectrogram,mel_spectrogram,amp

"""
"""
def get_intervals_dict(labelfile,map_idx):
    interval_dict = dict()
    with open(labelfile) as f:
        
        for i,row in enumerate(csv.reader(f, delimiter='\t')):
            if i == 0 or float(row[map_idx['dur']])< config.settings['thr']: continue
            wav   = row[map_idx['wav']]
            start = float(row[map_idx['start']])
            dur   = float(row[map_idx['dur']])
            end   = start+dur
            if wav not in interval_dict.keys():
                interval_dict[wav]=IntervalTree()
            interval_dict[wav].add(Interval(start,end))
    return interval_dict

"""
"""
def get_wavpaths(train_interval_dict,test_interval_dict):
    all_test_wavs_un = glob.glob(os.path.join(config.settings['test_data_source_path'],config.settings['wav_folder'],'*'))
    all_train_wavs_un = glob.glob(os.path.join(config.settings['train_data_source_path'],config.settings['wav_folder'],'*'))
    
    
    all_train_wavs  = []
    all_test_wavs    = []

    for wav_path in all_train_wavs_un:
        wav_name = os.path.basename(wav_path)
        if wav_name in train_interval_dict:
            all_train_wavs.append(wav_path)

    for wav_path in all_test_wavs_un:
        wav_name = os.path.basename(wav_path)
        if wav_name in test_interval_dict:
            all_test_wavs.append(wav_path)
        
    return all_train_wavs,all_test_wavs


"""
"""
class chunkaudio(object):
    
    def __init__(self,wav_path,interval_dict,window_size_t=2.5,target_sr=44100,pad_last=True,resample=True):
        
        self.wav_name = os.path.basename(wav_path)
        self.intervals = interval_dict[self.wav_name]
        self.orig_sr = librosa.get_samplerate(wav_path)
        self.target_sr = target_sr
        self.data, _= librosa.load(wav_path,self.orig_sr)
        self.window_size = int(window_size_t*self.orig_sr)
        excess = len(self.data)%self.window_size
        self.cum_stime =0
        self.curr_indx = 0
        self.win_no    = 0
        
        
        if excess and self.window_size < len(self.data) and pad_last:
            n_pad = self.window_size - excess
            self.data = np.pad(self.data,(0, n_pad),'reflect')
            
        if not resample:
            self.target_sr = self.orig_sr
            
    def __iter__(self):
        return self
            
    def __next__(self):
        data_chunk = self.data[self.curr_indx:self.curr_indx+self.window_size]
        chunk_stime = self.cum_stime
        chunk_etime = chunk_stime + len(data_chunk)/self.orig_sr
        
        if self.curr_indx >= len(self.data):
            raise StopIteration
            
        if self.intervals.overlaps(chunk_stime, chunk_etime):
            label = 1
        else:
            label = 0
        
        chunk_name = '_'.join([os.path.splitext(self.wav_name)[0],"%04d"%(self.win_no)])
            
        self.cum_stime = chunk_etime
        self.curr_indx = self.curr_indx+self.window_size
        self.win_no    +=1
        
        if self.target_sr != self.orig_sr:
            data_chunk = librosa.resample(data_chunk,self.orig_sr,self.target_sr)
        
        data_chunk     =   reduce_bg_noise(data_chunk)
        data_chunk     =  librosa.effects.preemphasis(data_chunk)
        return chunk_name,chunk_stime,chunk_etime,data_chunk,label



"""
"""
def process_data(wav_paths,window_size_t,interval_dict,target_sr,resample=True,pad_last=True,train=True):

    if train:
        print('\n\n\n####################### Processing Training Data ##################################')
    else:
        print('\n\n\n####################### Processing Testing Data ##################################')
    
    anno_data = []    
    for wav_path in wav_paths:
        print('processing file:{}'.format(os.path.basename(wav_path)))
        chk = chunkaudio(wav_path,interval_dict,window_size_t)
    
        for chunk_name,chunk_stime,chunk_etime,data_chunk,label in chk:
            D      = spectrogram(data_chunk,center=False,overlap=75)
            S      = mel_spectrogram(D=D,sr=target_sr,pwr=1)
            S_amp  = amp(S)
            plt.figure(figsize=(15,5))
            _=librosa.display.specshow(S_amp,cmap='viridis',y_axis='linear',sr=target_sr)
            plt.axis('off')
            plt.savefig(os.path.join(config.settings['data_path'],config.settings['spec_folder'],chunk_name+'.png'), bbox_inches='tight',pad_inches = 0)
            plt.tight_layout()
            plt.close()
            anno_data.append([str(os.path.join(config.settings['data_path'],config.settings['spec_folder'],chunk_name+'.png')),str(label)])
            print('chunk:{} is_orca={}'.format(chunk_name[-4:],label))
            

    if train:
        np.savetxt(os.path.join(config.settings['data_path'],'train.txt'),anno_data,fmt='%s')
    else:
        np.savetxt(os.path.join(config.settings['data_path'],'test.txt'),anno_data,fmt='%s')
    



if os.path.exists(config.settings['data_path']):
    shutil.rmtree(config.settings['data_path'])

os.makedirs(os.path.join(config.settings['data_path'],config.settings['spec_folder']))



train_interval_dict = get_intervals_dict(os.path.join(config.settings['train_data_source_path'],config.settings['train_annotation']),map_idx={'wav':0,'start':1,'dur':2})
test_interval_dict = get_intervals_dict(os.path.join(config.settings['test_data_source_path'],config.settings['test_annotation']),map_idx={'wav':0,'start':1,'dur':2})


all_train_wavs,all_test_wavs = get_wavpaths(train_interval_dict,test_interval_dict)



process_data(all_train_wavs,config.settings['window_size'],train_interval_dict,config.settings['samp_rate'],train=True)
process_data(all_test_wavs,config.settings['window_size'],test_interval_dict,config.settings['samp_rate'],train=False)

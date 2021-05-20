import os


settings={}


settings['root_path']                   = os.path.join(os.sep,'workspace')

settings['train_data_source_path']      = '../data_source/WHOIS09222019_PodCastRound1/train_data_09222019'
settings['test_data_source_path']       = '../data_source/OrcasoundLab09272017_Test/OrcasoundLab09272017_Test'

settings['train_annotation']      = 'train.tsv'
settings['test_annotation']       = 'test.tsv'
settings['wav_folder']            = 'wav'
settings['spec_folder']           = 'specs'



settings['data_path']  			  = '../data/round_1'
settings['train_anno_path']       = os.path.join(settings['data_path'],'train.txt')
settings['test_anno_path']        = os.path.join(settings['data_path'],'test.txt')

settings['samp_rate']             = 44100
settings['thr']                   = 0.1
settings['window_size']           = 2.45
settings['resample']              = True
settings['train_val_split_ratio'] = 0.9
settings['spec_resolution']		  = (300,200)


settings['num_epochs'] 			  = 50
settings['gpu']					  = 0
settings['batch_size']			  = 256
settings['lr']					  = 3e-4

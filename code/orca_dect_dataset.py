import os
import config 
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit



dataset_dir              = config.settings['data_path']
train_val_spec_paths     = np.loadtxt(os.path.join(config.settings['data_path'],'train.txt'),dtype='str')
X_train_val,y_train_val  = train_val_spec_paths[:,0],train_val_spec_paths[:,1]
sss                      = StratifiedShuffleSplit(n_splits=1, train_size=config.settings['train_val_split_ratio']) 

for train_idx, val_idx in sss.split(X_train_val, y_train_val):    
    X_train=X_train_val[train_idx]
    y_train=y_train_val[train_idx]
    X_val=X_train_val[val_idx]
    y_val=y_train_val[val_idx]


test_spec_paths          = np.loadtxt(os.path.join(config.settings['data_path'],'test.txt'),dtype='str')     
X_test,y_test            = test_spec_paths[:,0],test_spec_paths[:,1]






class OrcaDataset(Dataset):

    def __init__(self,X,y):
        self.X =X
        self.y =y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        img = Image.open(self.X[idx])
        label = int(self.y[idx])

        if img.mode != 'RGB':
            img = img.convert(mode='RGB')

        img = img.resize(config.settings['spec_resolution'])
        
        img_arr = np.array(img,dtype='float32')
        img_arr = img_arr/255.0

        img = transforms.ToTensor()(img_arr)
        return img,label



def get_train_dataset():
    return OrcaDataset(X_train,y_train)


def get_val_dataset():
    return OrcaDataset(X_val,y_val)

def get_test_dataset():
    return OrcaDataset(X_test,y_test)
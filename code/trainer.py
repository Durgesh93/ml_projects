import config
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from model import OrcaDectNet
import matplotlib.pyplot as plt
import orca_dect_dataset as dataset
from torch.utils.data import DataLoader
from tqdm import tqdm








train_ds  = dataset.get_train_dataset()
val_ds    = dataset.get_val_dataset()
test_ds   = dataset.get_test_dataset()
loss      = nn.CrossEntropyLoss()
net       = OrcaDectNet()
optimizer = optim.Adam(net.parameters(), lr=config.settings['lr'])
device    = torch.device("cuda:"+str(config.settings['gpu']) if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Running the code on GPU:{}'.format(torch.cuda.current_device()))

train_dl    = DataLoader(train_ds,batch_size=config.settings['batch_size'],shuffle=True,num_workers=4,drop_last=True)
val_dl      = DataLoader(val_ds,batch_size=config.settings['batch_size'],shuffle=False,num_workers=4,drop_last=False)
test_dl     = DataLoader(test_ds,batch_size=config.settings['batch_size'],shuffle=False,num_workers=4,drop_last=False)




net.to(device)


for epoch in range(0,config.settings['num_epochs']):

    epoch_train_loss     = []
    epoch_val_loss       = []
    epoch_train_acc      = []
    epoch_val_acc        = []


    for (batch_num, (X_batch,y_batch)) in enumerate(tqdm(train_dl)):


        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out   = net(X_batch)
        l     = loss(out, y_batch)
        l.backward()
        optimizer.step()


        epoch_train_loss.append(l.item())
        output  = torch.argmax(out,dim=-1).float()
        correct = (output == y_batch).float().sum()
        epoch_train_acc.append(correct.item()/X_batch.shape[0])
    


    for (batch_num, (X_batch,y_batch)) in enumerate(val_dl):
      
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        

        with torch.no_grad():
            out   = net(X_batch)
            l = loss(out, y_batch)
            epoch_val_loss.append(l.item())
            
            output  = torch.argmax(out,dim=-1).float()
            correct = (output == y_batch).float().sum()
            epoch_val_acc.append(correct.item()/X_batch.shape[0])

    print("Epoch:%2d,  Avg.Train Loss:%.4f, Avg. Val Loss: %.4f Avg. train acc:%2.2f Avg. val acc:%2.2f"% (epoch+1,np.average(epoch_train_loss),np.average(epoch_val_loss),np.average(epoch_train_acc),np.average(epoch_val_acc)))
    print('-----------------------------------------------------------------------------------------------')



















# Code for checking dataset.Please do not delete
# for (i, (X,y)) in enumerate(val_ds):
#   X = np.transpose(X.numpy(),(1,2,0))
#   X = (X*255).astype('uint8')
#   img = Image.fromarray(X)
#   img.save(str(i)+'.png')

#   if i > 5:
#       exit()

    
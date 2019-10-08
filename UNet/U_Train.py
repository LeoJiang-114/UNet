from UNet.U_Datasets import My_Datasets
from UNet.UNet_256 import Main_Net
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import os

img_path = r"F:\Datasets_Dispose\input\image"
lable_path = r"F:\Datasets_Dispose\input\label"
datasets = My_Datasets(img_path, lable_path)
train_data=data.DataLoader(dataset=datasets,batch_size=4,shuffle=True,drop_last=True)

net_path=r"F:\jkl\UNet\U_3Dimen.pth"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
net=Main_Net()
if os.path.exists(net_path):
    net=torch.load(net_path)
net=net.to(device)
optimizer=torch.optim.Adam(net.parameters())
loss_func=nn.BCELoss()

def Get_Loss(out,lable):
    ob_mask=lable[...,]>=0.8
    noob_mask=lable<0.8

    ob_loss=loss_func(out[ob_mask],noob_mask[ob_mask])
for epoch in range(1000):
    for i,(img,lable) in enumerate(train_data):
        img_data,lable_data=img.to(device),lable.to(device)
        out=net(img_data)

        loss=loss_func(out,lable_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10==0:
            accuracy=np.mean(out.cpu().detach().numpy()==lable.numpy())
            print("Loss: {} | Epoch: {} Accuracy:{} ".format(loss.item(),epoch,accuracy))
            torch.save(net,net_path)
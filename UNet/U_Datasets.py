import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image

class My_Datasets(data.Dataset):
    def __init__(self,img_path,lable_path):
        super(My_Datasets, self).__init__()
        self.img_path, self.lable_path=img_path,lable_path
        self.lable_list=os.listdir(lable_path)

    def __len__(self):
        return len(self.lable_list)

    def __getitem__(self, index):
        img_name_=self.lable_list[index].split("_mask")
        img_name=img_name_[0]+img_name_[1]
        #print(img_name)
        img_open=Image.open(os.path.join(self.img_path,img_name))
        img_data=np.array(img_open)/255
        img_data=torch.Tensor(img_data).permute(2,0,1)

        lable_open=Image.open(os.path.join(self.lable_path,self.lable_list[index]))#.convert("L")
        lable_data=np.array(lable_open)/255
        lable_data=torch.Tensor(lable_data).permute(2,0,1)#.reshape(1,256,256)
        #print(os.path.join(self.img_path,img_name),os.path.join(self.lable_path,self.lable_list[index]))

        return img_data,lable_data

if __name__ == '__main__':
    img_path=r"F:\Datasets_Dispose\input\image"
    lable_path=r"F:\Datasets_Dispose\input\label"
    datasets=My_Datasets(img_path,lable_path)
    img_data,lable_data=datasets[2]
    print(img_data.shape,lable_data.shape)

    # datasets = My_Datasets(img_path, lable_path)
    # train_data = data.DataLoader(dataset=datasets, batch_size=10, shuffle=True, drop_last=True)
    # for i, (img_data, lable_data) in enumerate(train_data):
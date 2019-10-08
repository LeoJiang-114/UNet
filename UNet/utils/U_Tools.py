import torch
import numpy

def Crop_torch(tensor,size):
    a=int((int(tensor.size(2))-size)/2)
    return tensor[:,:,a:a+size,a:a+size]


if __name__ == '__main__':
    a = torch.randn(1, 3, 568, 568)
    b=Crop_torch(a,392)
    print(b.shape)
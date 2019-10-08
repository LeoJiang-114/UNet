import torch
import numpy as np
from PIL import Image

#crop矩阵
# a=torch.randn(1,3,568,568)

# lable_path=r"F:\Datasets_Dispose\input\label\TCGA_CS_4941_19960909_11_mask.tif"
# img_open=Image.open(lable_path).convert("L")
# #img_open.show()
# img_data=np.array(img_open)
# print(img_data.shape)
# sum_value=np.sum(np.sum(img_data,axis=1))
# print(sum_value/255)#=1426,all=65536
# idxes=np.argmax(img_data,axis=1)
# print(idxes,idxes.shape)
# print(img_data[66,122])

# a=np.array([[1,2,3],
#             [3,4,5]])
# b=np.array([[1,2,3],
#             [3,5,4]])
# c=np.mean(a==b)
# print(c)

# a=torch.Tensor([[1.1,1.2],
#                 [2.4,2.6]])
# mask=a>1.5
# a[mask]=2
# print(a)

# import torch.nn as nn
#
# x=torch.Tensor([[0],[1],[7],[8]])
# Sig=nn.Sigmoid()
# for i in range(1000):
#     result=Sig(x)
#     x=result
# print(result)

# a=torch.Tensor([[[0,2,3],
#             [3,4,5]],
#             [[1, 2, 3],
#             [3, 4, 5]]])
# b=torch.Tensor([[[0,2,3],
#             [3,4,5]],
#             [[1, 2, 3],
#             [3, 4, 5]]])
#
# c=torch.cat((a,b),dim=1)
# print(c)

# a='TCGA_CS_4941_19960909_10_mask.tif'
# b=a.split("_mask")
# print(b[0]+b[1])

a=torch.Tensor([[1,2,3],
                [4,5,6],
                [7,8,9]])
ob_mask=a[...,]>3
print(ob_mask)
a[ob_mask]=0
print(a)
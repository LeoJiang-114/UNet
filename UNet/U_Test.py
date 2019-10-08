import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2 as cv

net_path=r"F:\jkl\UNet\Net_Save\U_net_1000.pth"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
net=torch.load(net_path).to(device)

img_path=r"F:\MRISegmentation_FlyAI\data\input\image"
save_path=r"F:\Datasets_Test\U_mask"
if not os.path.exists(save_path):
    os.makedirs(save_path)
for img_name in os.listdir(img_path):
    img_open = Image.open(os.path.join(img_path,img_name))
    img_data = np.array(img_open) / 255
    img_data = torch.Tensor([img_data.transpose(2, 0, 1)]).to(device)

    out_ = net(img_data)
    # print(out_.shape,out_)
    out=out_[0].cpu().detach()
    # ob_mask=out>0.5
    # out[ob_mask]=1.0

    out = np.array(out, dtype=np.uint8)
    out = np.transpose(out,(1,2,0))* 255
    # print(out.shape)

    cv.imwrite(r"{}\{}".format(save_path,img_name),out)
    # img_back = Image.fromarray(out)
    # img_back.save(r"{}\{}".format(save_path,img_name))
    # img_back.show()

    # plt.imshow(out)
    # plt.pause(1)

    # cv.imshow(" ", out)
    # cv.waitKey(1000)
    # cv.destroyAllWindows()


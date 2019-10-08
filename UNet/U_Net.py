import torch
import torch.nn as nn
import torch.nn.functional as F
import UNet.utils.U_Tools as tools

class ConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        output=self.layer(x)
        return output

class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride, padding):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        output=self.layer(x)
        return output

class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        output=self.layer(x)
        return F.interpolate(output,scale_factor=2,mode="nearest")

class Main(nn.Module):
    def __init__(self):
        super(Main, self).__init__()
        self.UL_First = ConvLayer(3, 64, 3)
        self.Downsample_First = Downsample(64,64,3,2,1)

        self.UL_Second = ConvLayer(64, 128, 3)
        self.Downsample_Second = Downsample(128, 128, 3, 2, 1)

        self.UL_Third = ConvLayer(128, 256, 3)
        self.Downsample_Third = Downsample(256, 256, 3, 2, 1)

        self.UL_Forth = ConvLayer(256, 512, 3)
        self.Downsample_Forth = Downsample(512, 512, 3, 2, 1)

        self.UL_Fifth = ConvLayer(512, 1024, 3)

        self.Upsample_First = Upsample(1024, 512)
        self.UR_First = ConvLayer(1024, 512, 3)

        self.Upsample_Second = Upsample(512, 256)
        self.UR_Second = ConvLayer(512, 256, 3)

        self.Upsample_Third = Upsample(256,128)
        self.UR_Third = ConvLayer(256, 128, 3)

        self.Upsample_Forth = Upsample(128, 64)
        self.UR_Forth = ConvLayer(128, 64, 3)

        self.last_layer=nn.Sequential(
            nn.Conv2d(64,2,3,1,1)
        )


    def forward(self, x):
        ul_first=self.UL_First(x)
        crop_first=tools.Crop_torch(ul_first,392)
        ul_first=self.Downsample_First(ul_first)

        ul_second=self.UL_Second(ul_first)
        crop_second = tools.Crop_torch(ul_second, 200)
        ul_second=self.Downsample_Second(ul_second)

        ul_third=self.UL_Third(ul_second)
        crop_third = tools.Crop_torch(ul_third, 104)
        ul_third=self.Downsample_Third(ul_third)

        ul_forth=self.UL_Forth(ul_third)
        crop_forth = tools.Crop_torch(ul_forth, 56)
        ul_forth=self.Downsample_Forth(ul_forth)

        ul_fifth=self.UL_Fifth(ul_forth)

        up_first=self.Upsample_First(ul_fifth)
        # print(up_first.shape,crop_forth.shape)
        cat_first=torch.cat((up_first,crop_forth),dim=1)
        ur_first=self.UR_First(cat_first)

        up_second=self.Upsample_Second(ur_first)
        cat_second=torch.cat((up_second,crop_third),dim=1)
        ur_second=self.UR_Second(cat_second)

        up_third=self.Upsample_Third(ur_second)
        cat_third=torch.cat((up_third,crop_second),dim=1)
        ur_third=self.UR_Third(cat_third)

        up_forth=self.Upsample_Forth(ur_third)
        cat_forth=torch.cat((up_forth,crop_first),dim=1)
        ur_forth=self.UR_Forth(cat_forth)

        output=self.last_layer(ur_forth)

        return output

if __name__ == '__main__':
    x=torch.randn(2,3,572,572)
    net=Main()
    output=net(x)
    print(output.shape)
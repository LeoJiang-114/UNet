import torch
import torch.nn as nn
import torch.nn.functional as F
import UNet.utils.U_Tools as tools

class Conv_Net(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding):
        super(Conv_Net, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(),
        )

    def forward(self, x):
        result=self.layer(x)
        return result

class Down_Sample(nn.Module):
    def __init__(self,channel):
        super(Down_Sample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=2,stride=2),
            nn.BatchNorm2d(channel),
            nn.PReLU()
        )
    def forward(self, x):
        result=self.layer(x)
        return result

class Up_Sample(nn.Module):
    def __init__(self,channel):
        super(Up_Sample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel//2, 1, 1),
            nn.BatchNorm2d(channel//2),
            nn.PReLU()
        )
    def forward(self, x):
        result=self.layer(x)
        return F.interpolate(result,scale_factor=2,mode="nearest")

class Main_Net(nn.Module):

    def __init__(self):
        super(Main_Net, self).__init__()
        self.Left_C1=Conv_Net(3,64,3,1,1)
        self.Down1=Down_Sample(64)#128*128

        self.Left_C2 = Conv_Net(64, 128, 3, 1, 1)
        self.Down2 = Down_Sample(128)  # 64*64

        self.Left_C3 = Conv_Net(128, 256, 3, 1, 1)
        self.Down3 = Down_Sample(256)  # 32*32

        self.Left_C4 = Conv_Net(256, 512, 3, 1, 1)
        self.Down4 = Down_Sample(512)  # 16*16

        self.Left_C5 = Conv_Net(512, 1024, 3, 1, 1)#16*16

        self.Up1=Up_Sample(1024)#32*32
        self.Right_C1=Conv_Net(1024,512,3,1,1)

        self.Up2 = Up_Sample(512)
        self.Right_C2 = Conv_Net(512,256,3,1,1)# 64*64

        self.Up3=Up_Sample(256)
        self.Right_C3 = Conv_Net(256,128,3,1,1)# 128*128

        self.Up4=Up_Sample(128)
        self.Right_C4 = Conv_Net(128,64,3,1,1)  # 256*256

        self.Last_layer=nn.Sequential(
            nn.Conv2d(64,1,1,1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        left_c1 = self.Left_C1(x)
        down1 = self.Down1(left_c1)#128*128

        left_c2 = self.Left_C2(down1)
        down2 = self.Down2(left_c2)#64*64

        left_c3 = self.Left_C3(down2)
        down3 = self.Down3(left_c3)#32*32

        left_c4 = self.Left_C4(down3)
        down4 = self.Down4(left_c4)#16*16

        left_c5=self.Left_C5(down4)
        #print(left_c5.shape)

        up1=self.Up1(left_c5)
        #print(up1.shape,left_c4.shape)
        cat1=torch.cat((up1,left_c4),dim=1)
        right_c1=self.Right_C1(cat1)

        up2=self.Up2(right_c1)
        cat2 = torch.cat((up2, left_c3), dim=1)
        right_c2=self.Right_C2(cat2)

        up3 = self.Up3(right_c2)
        cat3 = torch.cat((up3, left_c2), dim=1)
        right_c3 = self.Right_C3(cat3)

        up4 = self.Up4(right_c3)
        cat4 = torch.cat((up4, left_c1), dim=1)
        right_c4 = self.Right_C4(cat4)

        last=self.Last_layer(right_c4)
        return last

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=Main_Net()
    result=net(x)
    print(result.shape)

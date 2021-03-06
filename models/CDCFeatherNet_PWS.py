# the code base on https://github.com/tonylins/pytorch-mobilenet-v2
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import sys
sys.path.append("..")
# from torchsummary import summary
from tools.benchmark import compute_speed, stat

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=0.7),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 1, 1, 0, bias=False, theta=0.7),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class PixelMapEstimator(nn.Module):
    def __init__(self, in_chn,num_class):  
        super(PixelMapEstimator, self).__init__()
        self.in_chn = in_chn
        self.num_class = num_class
        self.pwblock = nn.Sequential(

                nn.Conv2d(self.in_chn , self.in_chn, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.in_chn),
                nn.ReLU(inplace=True),

                # dw
                nn.Conv2d(self.in_chn , self.num_class, 1, 1, 0, bias=False)
            )
    
    def forward(self,x):
        return self.pwblock(x)

class PixelMapEstimator_BN(nn.Module):
    def __init__(self, in_chn,num_class):  
        super(PixelMapEstimator_BN, self).__init__()
        self.in_chn = in_chn
        self.num_class = num_class
        self.pwblock = nn.Sequential(

                nn.Conv2d(self.in_chn , self.in_chn, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.in_chn),
                nn.ReLU(inplace=True),

                # dw
                nn.Conv2d(self.in_chn , self.num_class, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.num_class),
                nn.ReLU(inplace=True) 
            )
    
    def forward(self,x):
        return self.pwblock(x)

class PixelMapEstimator_OBN(nn.Module):
    def __init__(self, in_chn,num_class):  
        super(PixelMapEstimator_OBN, self).__init__()
        self.in_chn = in_chn
        self.num_class = num_class
        self.pwblock = nn.Sequential(

                nn.Conv2d(self.in_chn , self.in_chn, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.in_chn),
                nn.ReLU(inplace=True),

                # dw
                nn.Conv2d(self.in_chn , self.num_class, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.num_class)
            )
    
    def forward(self,x):
        return self.pwblock(x)

# reference form : https://github.com/moskomule/senet.pytorch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y
        return x.mul(y)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, downsample=None, theta=0.7):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d_cd(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, theta=theta),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),#???????????????
                # pw-linear
                Conv2d_cd(hidden_dim, oup, 1, 1, 0, bias=False, theta=theta),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d_cd(inp, hidden_dim, 1, 1, 0, bias=False,theta=theta),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),#?????????
                # dw
                Conv2d_cd(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False,theta=theta),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                Conv2d_cd(hidden_dim, oup, 1, 1, 0, bias=False,theta=theta),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            if self.downsample is not None:
                return self.downsample(x) + self.conv(x)
            else:
                return self.conv(x)


class CDCFeatherNet_PWS(nn.Module):
    def __init__(self, n_class=2, img_channel=1,input_size=112, se=False, avgdown=False, width_mult=1.0, theta=0.7):
        super(CDCFeatherNet_PWS, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.avgdown = avgdown
        self.img_channel = img_channel
        self.width_mult = width_mult
        interverted_residual_setting = [

            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2],  # 56x56
            [6, 48, 6, 2],  # 14x14
            [6, 64, 3, 2],  # 7x7
        ]

        # building first layer
        assert input_size % 16 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(img_channel, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown and s != 1:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                                                   nn.BatchNorm2d(input_channel),
                                                   Conv2d_cd(input_channel, output_channe+l, kernel_size=1, bias=False,theta=theta))
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample=downsample,theta=theta))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample=downsample,theta=theta))
                input_channel = output_channel
            if self.se:
                self.features.append(SELayer(input_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        #         building last several layers
        self.final_DW = nn.Sequential(Conv2d_cd(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
                                                groups=input_channel, bias=False,theta=theta))

        self._initialize_weights()

    def forward(self, x):#???????????????
        x = self.features[:-10](x)
        ftmap = self.features[-10](x)
        # print("===========ftmap========: ", ftmap.shape)
        x = self.features[-9:](ftmap)
        x = self.final_DW(x)

        x = x.view(x.size(0), -1)
        return x,ftmap

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def CDCFeatherNetA_PWS(se=False, width_mult=1.0,theta=0.7):
    model = CDCFeatherNet_PWS(se = se, width_mult=width_mult, theta=theta)
    return model

def CDCFeatherNetB_PWS(se=True, img_channel=1, width_mult=1.0):
    model = CDCFeatherNet_PWS(se=se, avgdown=True, img_channel=img_channel, width_mult=width_mult)
    return model


# if __name__ == "__main__":
#     # model = FaceFeatherNetB_v2()         # Total Flops(Conv Only): 70.46MFlops, model size = 1.36MB
#     model = FaceFeatherNetA_PWS(se=False)  # Total Flops(Conv Only): 70.46MFlops, model size = 1.35MB
#     print(model)
#     rand_input = torch.rand([64, 1, 112, 112])
#     y, f = model(rand_input)
#     print("==========y==========: ", y.shape)
#     print("==========f==========: ", f.shape)
#     estimator = PixelMapEstimator(32, 2)
#     fm = estimator(f)
#     print("======estimator=======: ", fm.shape)

#     # str_input_size = '1x1x112x112'
#     # input_size = tuple(int(x) for x in str_input_size.split('x'))
#     # stat(model, input_size)
# #
# #     # summary(model, (1, 112, 112))7



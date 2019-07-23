import torch
import torch.nn.functional as F
import torch.nn as nn

__all__ = ['RFCN']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, recurrent_n=2, active_func=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(in_planes, planes),
            nn.BatchNorm2d(planes),
            active_func,
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
            active_func
        )

    def forward(self, x):
        return self.block(x)


class RecurrentBlock(nn.Module):
    def __init__(self, in_planes, planes, recurrent_n=2, active_func=nn.ReLU(inplace=True)):
        super(RecurrentBlock, self).__init__()
        self.recurrent_n = recurrent_n
        self.conv1 = conv3x3(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.active_func = active_func

        self.RCL = conv3x3(planes, planes)
        self.bn2 = nn.ModuleList()
        for n in range(recurrent_n):
            self.bn2.append(nn.BatchNorm2d(planes))

    def forward(self, x):
        conv1 = self.bn1(self.conv1(x))
        active = self.active_func(conv1)
        for n in range(self.recurrent_n):
            conv2 = self.bn2[n](self.RCL(active))
            active = self.active_func(conv1 + conv2)

        return active


class RecurrentBlock2(nn.Module):
    def __init__(self, in_planes, planes, recurrent_n=2, active_func=nn.ReLU(inplace=True)):
        super(RecurrentBlock2, self).__init__()
        self.recurrent_basic_block1 = RecurrentBlock(in_planes, planes, recurrent_n=recurrent_n, active_func=active_func)
        self.recurrent_basic_block2 = RecurrentBlock(planes, planes, recurrent_n=recurrent_n, active_func=active_func)

    def forward(self, x):
        out = self.recurrent_basic_block1(x)
        out = self.recurrent_basic_block2(out)
        return out


class RecurrentBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, recurrent_n=2, active_func=nn.ReLU(inplace=True)):
        super(RecurrentBasicBlock, self).__init__()
        self.recurrent_n = recurrent_n
        self.recurrent_basic_block_1 = BasicBlock(in_planes, planes, recurrent_n=recurrent_n, active_func=active_func)
        self.recurrent_basic_block_2 = BasicBlock(planes, planes, recurrent_n=recurrent_n, active_func=active_func)

    def forward(self, x):
        out = self.recurrent_basic_block_1(x)

        out_2 = out
        for n in range(self.recurrent_n):
            out_2 = self.recurrent_basic_block_2(out_2) + out

        return out_2


class RecurrentBasicBlock2(nn.Module):
    def __init__(self, in_planes, planes, stride=1, recurrent_n=2, active_func=nn.ReLU(inplace=True)):
        super(RecurrentBasicBlock2, self).__init__()
        self.recurrent_n = recurrent_n
        self.recurrent_basic_block_1 = RecurrentBasicBlock(in_planes, planes, recurrent_n=recurrent_n, active_func=active_func)
        self.recurrent_basic_block_2 = RecurrentBasicBlock(planes, planes, recurrent_n=recurrent_n, active_func=active_func)

    def forward(self, x):
        out = self.recurrent_basic_block_1(x)
        out = self.recurrent_basic_block_2(out)
        return out


class BasicNet(nn.Module):

    def __init__(self, filters, block, recurrent_n=2, layers=4, kernel=3, inplanes=3, num_classes=2, active_func=None):
        super().__init__()
        self.filters = filters
        self.layers = layers
        self.kernel = kernel
        self.inplanes = inplanes
        self.active_func = active_func

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.down_conv_list = nn.ModuleList()
        self.down_conv_2_list = nn.ModuleList()
        self.down_scale_conv_list = nn.ModuleList()
        for i in range(layers):
            self.down_conv_list.append(block(inplanes if i == 0 else filters * 2 ** i, filters * 2 ** i,
                                             recurrent_n=recurrent_n, active_func=self.leakyRelu))
            self.down_conv_2_list.append(nn.Sequential(
                conv3x3(filters * 2 ** i, filters * 2 ** (i + 1), 2),
                nn.BatchNorm2d(filters * 2 ** (i + 1))
            ))
            if i != layers - 1:
                self.down_scale_conv_list.append(nn.Sequential(
                    conv3x3(inplanes, filters * 2 ** (i + 1)),
                    nn.BatchNorm2d(filters * 2 ** (i + 1))
                ))

        self.bottom = block(filters * 2 ** layers, filters * 2 ** layers, recurrent_n=recurrent_n, active_func=self.leakyRelu)

        self.up_convtrans_list = nn.ModuleList()
        self.up_conv_list = nn.ModuleList()
        self.out_conv = nn.ModuleList()
        for i in range(0, layers):
            self.up_convtrans_list.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters * 2 ** (layers - i), out_channels=filters * 2 ** max(0, layers - i - 1),
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.BatchNorm2d(filters * 2 ** max(0, layers - i - 1))
            ))
            self.up_conv_list.append(block(filters * 2 ** max(0, layers - i - 1), filters * 2 ** max(0, layers - i - 1),
                                           recurrent_n=recurrent_n, active_func=self.relu))

            self.out_conv.append(conv3x3(filters * 2 ** (self.layers - i - 1), num_classes))

    def forward(self, x):
        x_down = down_out = x
        down_outs = []
        for i in range(0, self.layers):
            down_out = self.down_conv_list[i](down_out)
            down_outs.append(down_out)
            down_out = self.down_conv_2_list[i](down_out)
            if i != self.layers - 1:
                x_down = F.avg_pool2d(x_down, 2)
                down_out = self.leakyRelu(down_out + self.down_scale_conv_list[i](x_down))
            else:
                down_out = self.leakyRelu(down_out)

        # bottom branch
        up_out = self.bottom(down_out)

        outs = []
        out_final = None
        for j in range(self.layers):
            up_out = self.relu(down_outs[self.layers - j - 1] + self.up_convtrans_list[j](up_out))
            up_out = self.up_conv_list[j](up_out)

            out = F.interpolate(up_out, size=(up_out.shape[2] * 2 ** (self.layers - j - 1), up_out.shape[3] * 2 ** (self.layers - j - 1)))
            out = self.out_conv[j](out)
            if self.active_func is not None:
                out = self.active_func(out, dim=1)
            outs.append(out)
            out_final = out if out_final is None else out_final + out

        out_final /= self.layers
        outs.append(out_final)
        return outs


def RFCN(filters, layers=4, kernel=3, inplanes=3, num_classes=2, active_func=None, recurrent_n=2):
    return BasicNet(filters=filters, block=RecurrentBasicBlock2, layers=layers, kernel=kernel, inplanes=inplanes, num_classes=num_classes,
                    active_func=active_func, recurrent_n=recurrent_n)

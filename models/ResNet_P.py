import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


'''
downsampling
'''
class Downsampling(nn.Module):
    def __init__(self, ch_in, d):

        super(Downsampling, self).__init__()
        d += 1
        self.avp33 = nn.AvgPool2d(kernel_size=(3, 3), stride=(d, d), padding=1)
        self.conv11 = nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch_in)
        self.conv33 = nn.Conv2d(ch_in, ch_in, kernel_size=(3, 3), stride=(d, d), padding=1, bias=False)
        self.bn_ = nn.BatchNorm2d(ch_in)

        self.conv11_out = nn.Conv2d(2*ch_in, 2*ch_in, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn_out = nn.BatchNorm2d(2*ch_in)

    def forward(self, x):
        p1 = F.relu(self.bn(self.conv11(self.avp33(x))))
        p2 = F.relu(self.bn_(self.conv33(x)))
        down_out = F.relu(self.bn_out(self.conv11_out(torch.cat((p1, p2), dim=1))))
        return down_out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class make_partial(nn.Module):
    def __init__(self, block, depth, ch_in, p):
        super(make_partial, self).__init__()
        self.p = p
        self.depth = depth // p

        self.branches = nn.ModuleList()
        for i in range(p):
            branch = self._make_layer(block, ch_in, self.depth, stride=1)
            self.branches.append(branch)

        self.fusion = Bottleneck(ch_in=ch_in * p)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        splitted_parts = x.chunk(self.p, dim=1)
        outputs = []
        for idx in range(len(splitted_parts)):
            outputs.append(self.branches[idx](splitted_parts[idx]))
        cat = torch.cat(outputs, dim=1)

        return self.fusion(cat)


'''
Bottleneck layer(fusion)
'''
class Bottleneck(nn.Module):
    def __init__(self, ch_in):
        super(Bottleneck, self).__init__()
        self.conv11 = nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1), padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch_in)

    def forward(self, x):
        x = F.relu(self.bn(self.conv11(x)))
        return x


'''
get 1*1 conv out ch
'''
def get_Bottleneck_out(ch_in, seita, p):
    ch_out = int(ch_in * seita)
    ch_out = int((ch_out // (p / 2)) * (p / 2))  # 因为包含下采样层
    return ch_out


class ResNet(nn.Module):
    def __init__(self, num_classes, deep_list, p, w):
        super(ResNet, self).__init__()
        base_width = 16

        ch = (p * base_width * w) // 2
        self.conv_in = nn.Conv2d(3, ch, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_in = nn.BatchNorm2d(ch)

        self.down_stem = Downsampling(ch_in=ch, d=0)
        ch = (base_width * w)
        self.stage1 = self._make_layer(ch_in=ch, depth=deep_list[0], p=p)
        ch = (base_width * w) * p
        self.down_1 = Downsampling(ch_in=ch, d=1)
        ch = (base_width * w) * 2
        self.stage2 = self._make_layer(ch_in=ch, p=p, depth=deep_list[1])
        ch = (base_width * w) * 2 * p
        self.down_2 = Downsampling(ch_in=ch, d=1)
        ch = (base_width * w) * 4
        self.stage3 = self._make_layer(ch_in=ch, p=p, depth=deep_list[2])

        self.linear = nn.Linear(ch * p, num_classes)

        self.init_weights()

    def _make_layer(self, ch_in, depth, p):
        layers = []
        layers.append(make_partial(block=BasicBlock, depth=depth, ch_in=ch_in, p=p))

        return nn.Sequential(*layers)

        # KaiMing Initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.down_stem(x)
        x = self.stage1(x)
        x = self.down_1(x)
        x = self.stage2(x)
        x = self.down_2(x)
        x = self.stage3(x)

        out = F.avg_pool2d(x, x.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet32(num_classes, w, p):
    return ResNet(num_classes=num_classes, deep_list=[4, 4, 4], p=p, w=w)


def resnet56(num_classes, w, p):
    return ResNet(num_classes=num_classes, deep_list=[8, 8, 8], p=p, w=w)


def resnet110(num_classes, w, p):
    return ResNet(num_classes=num_classes, deep_list=[16, 16, 16], p=p, w=w)


def resnet152(num_classes, w, p):
    return ResNet(num_classes=num_classes, deep_list=[24, 24, 24], p=p, w=w)


if __name__ == '__main__':
    '''
    test
    '''

    # Flops
    from torchstat import stat
    model = resnet32(num_classes=100, w=1, p=2).eval()
    print(model)
    stat(model, (3, 32, 32))
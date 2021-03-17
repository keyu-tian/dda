from collections import OrderedDict
import torch
import torch.nn as nn

_bn_mom = 0.99


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, t, stride=1, act=nn.ReLU6):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes * t, momentum=_bn_mom)
        self.conv2 = nn.Conv1d(inplanes * t, inplanes * t, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes * t)
        self.bn2 = nn.BatchNorm1d(inplanes * t, momentum=_bn_mom)
        self.conv3 = nn.Conv1d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(outplanes, momentum=_bn_mom)
        self.act = act(inplace=True)

        self.with_skip_connect = stride == 1 and inplanes == outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.with_skip_connect:
            out += residual

        return out


class MBV2(nn.Module):

    def __init__(self, input_size, num_classes,
                 bn0=False,
                 bn_mom=0.9,
                 scale=1.0,
                 t=[0, 1, 6, 6, 6, 6, 6, 6],
                 n=[1, 1, 2, 3, 4, 3, 3, 1],
                 c=[32, 16, 24, 32, 64, 96, 160, 320],
                 dropout=0.2
                 ):

        super(MBV2, self).__init__()

        global _bn_mom
        _bn_mom = bn_mom
        if bn0:
            self.bn0 = nn.BatchNorm1d(1, momentum=bn_mom)
        else:
            self.bn0 = None

        self.act = nn.ReLU6(inplace=True)
        self.num_classes = num_classes

        self.c = [_make_divisible(ch * scale, 8) for ch in c]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.t = t
        self.n = n
        assert self.t[0] == 0

        self.conv1 = nn.Conv1d(1, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm1d(self.c[0], momentum=_bn_mom)

        self.main = self._make_all()

        # Last convolution has 720 output channels for scale <= 1
        last_ch = 640 if scale < 1 else _make_divisible(640 * scale, 8)
        self.conv_last = nn.Conv1d(self.c[-1], last_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm1d(last_ch, momentum=_bn_mom)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        if dropout > 1e-6:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.feature_dim = last_ch
        self.classifier = nn.Linear(last_ch, self.num_classes)

    def _make_stage(self, inplanes, outplanes, n, stride, t):
        modules = OrderedDict()
        modules['MBConv_0'] = Bottleneck(inplanes, outplanes, t, stride=stride)
        for i in range(1, n):
            modules['MBConv_{}'.format(i)] = Bottleneck(outplanes, outplanes, t, stride=1)
        return nn.Sequential(modules)

    def _make_all(self):
        modules = OrderedDict()
        for i in range(1, len(self.c)):
            modules['stage_{}'.format(i)] = self._make_stage(inplanes=self.c[i-1], outplanes=self.c[i],
                                                             n=self.n[i], stride=self.s[i], t=self.t[i])
        return nn.Sequential(modules)

    def forward(self, x, returns_feature=False):
        if self.bn0 is not None:
            x = self.bn0(x)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.main(x)

        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.act(x)

        x = self.avgpool(x)
        if returns_feature:
            return x.view(x.size(0), -1)
        else:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

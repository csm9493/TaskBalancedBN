from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
        return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
  Except it puts the stride on 3x3 conv when available.
  """

    def __init__(self, cin, cout=None, cmid=None, stride=1, norm_layer = None):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4
        
        if isinstance(norm_layer, AdaptiveSplitRepeatBN):

            self.gn1 = norm_layer(cin)
            self.conv1 = conv1x1(cin, cmid)
            self.gn2 = norm_layer(cmid)
            self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
            self.gn3 = norm_layer(cmid)
            self.conv3 = conv1x1(cmid, cout)
            self.relu = nn.ReLU(inplace=True)
            
        else:
            
            self.gn1 = nn.GroupNorm(32, cin)
            self.conv1 = conv1x1(cin, cmid)
            self.gn2 = nn.GroupNorm(32, cmid)
            self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
            self.gn3 = nn.GroupNorm(32, cmid)
            self.conv3 = conv1x1(cmid, cout)
            self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
          # Projection also with pre-activation according to paper.
          self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=''):
        convname = 'standardized_conv2d'
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
            self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
            self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
            self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
            self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
            self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
            self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
            if hasattr(self, 'downsample'):
                w = weights[f'{prefix}a/proj/{convname}/kernel']
                self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units = [2, 2, 2, 2], width_factor = 1, head_size=21843, zero_head=False, norm_layer = None):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
            ))),
        ]))
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        
        if norm_layer != None:
            
            print ('ResNet V2 with AdaptiveSplitRepeatBN')

            self.head = nn.Sequential(OrderedDict([
                ('gn', norm_layer(2048*wf)),
                ('relu', nn.ReLU(inplace=True)),
                ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
                ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
            ]))
            
        else:
            
            self.head = nn.Sequential(OrderedDict([
                ('gn', nn.GroupNorm(32, 2048*wf)),
                ('relu', nn.ReLU(inplace=True)),
                ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
                ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
            ]))
            
        self.fc = nn.Linear(21843, 1000)
        self.head_var = 'fc'
        
    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        x = self.head(x)[...,0,0]
        x = self.fc(x)
#         assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
        if self.zero_head:
            nn.init.zeros_(self.head.conv.weight)
            nn.init.zeros_(self.head.conv.bias)
        else:
            self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
            self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

        for bname, block in self.body.named_children():
            for uname, unit in block.named_children():
                unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')
                
def resnet18_v2(pretrained=False, norm_layer = None):
    if pretrained:
        raise NotImplementedError
    # change n=3 for ResNet-20, and n=9 for ResNet-56
    model = ResNetV2([2,2,2,2], 1, norm_layer = norm_layer)
    return model

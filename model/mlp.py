from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn

# from utils import get_af, init_params


class FCBlock(nn.Module):
    def __init__(self, ind, oud, bn_mom):
        super(FCBlock, self).__init__()
        self.linear = nn.Linear(ind, oud, bias=False)
        self.bn = nn.BatchNorm1d(oud, momentum=bn_mom)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, num_classes,
                 bn0=False,
                 bn0_flatten=False,
                 bn_mom=0.9,
                 base_hidden_dim=120,
                 num_layers=4,
                 dropout_p=None,
                 ):
        super(MLP, self).__init__()
        hid_dims = [base_hidden_dim * i for i in range(num_layers, 0, -1)]
        relief_dim = round((input_size + hid_dims[0]) / 2)
        hid_dims.insert(0, relief_dim)
        
        self.bn_mom = bn_mom
        self.using_dropout = dropout_p is not None and dropout_p > 1e-6
        
        self.bn0_flatten = bn0_flatten
        if bn0:
            self.bn0 = nn.BatchNorm1d(input_size if bn0_flatten else 1, momentum=bn_mom)
        else:
            self.bn0 = None
        self.backbone = self._make_backbone(
            len(hid_dims),
            [input_size] + hid_dims[:-1],
            hid_dims
        )
        if self.using_dropout:
            self.dropout_p = dropout_p
            self.dropout = nn.Dropout(p=dropout_p)
        self.feature_dim = hid_dims[-1]
        self.classifier = nn.Linear(hid_dims[-1], num_classes, bias=True)
    
    def forward(self, x, returns_feature=False):
        if self.bn0_flatten:
            flatten_x = x.view(x.shape[0], -1)
            if self.bn0 is not None:
                flatten_x = self.bn0(flatten_x)
        else:
            if self.bn0 is not None:
                x = self.bn0(x)
            flatten_x = x.view(x.shape[0], -1)
            
        feature = self.backbone(flatten_x)
        if returns_feature:
            return feature
        
        if self.using_dropout:
            feature = self.dropout(feature)
        logits = self.classifier(feature)
        return logits
    
    def _make_backbone(self, num_layers, in_dims, out_dims):
        backbone = OrderedDict({
            f'linear_{i}': FCBlock(ind=in_dims[i], oud=out_dims[i], bn_mom=self.bn_mom)
            for i in range(num_layers)
        })
        return nn.Sequential(backbone)

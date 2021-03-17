import torch

from utils.misc import init_params
from model.lstm import LSTM
from model.lstm_fcn import LSTM_FCN
from model.mlp import MLP
from model.mobilenet_v2 import MBV2
from model.resnet import Res18


def model_entry(model_cfg) -> torch.nn.Module:
    return globals()[model_cfg.name](**model_cfg.kwargs)


if __name__ == '__main__':
    bs, sig_len = 16, 31
    num_classes = 5
    for bn0 in [True, False]:
        nets = [
            LSTM(input_size=sig_len, num_classes=num_classes, batch_size=bs, dropout=0.1, bn0=bn0),
            Res18(input_size=sig_len, num_classes=num_classes, bn0=bn0),
            LSTM_FCN(input_size=sig_len, num_classes=num_classes, hidden_dim=32),
            MLP(input_size=sig_len, num_classes=num_classes, dropout_p=0.2, base_hidden_dim=200, num_layers=4, bn0=bn0),
            MBV2(input_size=sig_len, num_classes=num_classes, dropout=0, scale=0.5, bn0=True)
        ]
        
        for net in nets:
            init_params(net)
            inp = torch.rand(bs, 1, sig_len)
            params = sum([p.numel() for p in net.parameters()]) / 1e6
            
            oup = net(inp, returns_feature=False)
            assert tuple(oup.shape) == (bs, num_classes)
            oup = net(inp, returns_feature=True)
            assert oup.shape[-1] == net.feature_dim
            
            print(f'{net.__class__.__name__:10s}: bn0={bn0}\t, params={params:5.2f}M, oup.shape={oup.shape}')
        print('')

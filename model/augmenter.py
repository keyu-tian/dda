import torch
import torch.nn as nn

from model import MLP


class Augmenter(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = MLP(input_size=feature_dim, num_classes=2, bn0=False, base_hidden_dim=200, num_layers=3)
        torch.nn.init.normal_(self.fc.classifier.weight, std=0.001)
        torch.nn.init.constant_(self.fc.classifier.bias, -3)    # (-3).sigmoid ~ 0.05

    def forward(self, fea):
        # alpha: (bs, 2)
        alpha = self.fc(fea).sigmoid() / 5
        return alpha   # [0, 0.2]


if __name__ == '__main__':  # testing
    bs, fea_dim = 32, 512
    aa: MLP = Augmenter(feature_dim=fea_dim)
    print(aa.fc)
    
    oup = aa(torch.rand(bs, fea_dim))
    print(f'output.shape: {oup.shape}')
    
    s = 0
    for p in aa.parameters():
        s += p.numel()
    print(f'num_params: {s/1e6:.3f} M')


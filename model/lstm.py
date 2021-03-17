import torch
import torch.nn as nn


class LSTM(nn.Module):
    
    def __init__(self, batch_size, input_size, num_classes,
                 bn0=False,
                 bn_mom=0.9,
                 in_dim=1,
                 hidden_dim=200,
                 num_layers=4,
                 dropout=0.2,
                 lstm_dropout=0.8,
                 bidirectional=True
                 ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers

        if bn0:
            self.bn0 = nn.BatchNorm1d(1, momentum=bn_mom)
        else:
            self.bn0 = None
        
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        
        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim * self.num_dir, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_mom),
            nn.ReLU(True),
        )
        self.feature_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_mom),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def init_hidden(self):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim).cuda()
            c0 = torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim)
            c0 = torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim)
        return h0, c0
    
    def forward(self, x, returns_feature=False):  # x is (batch_size, 1, inp_sz), permute to (inp_sz, batch_size, 1)
        if self.bn0 is not None:
            x = self.bn0(x)
        x = x.permute(2, 0, 1)
        lstm_out, (h, c) = self.lstm(x, self.init_hidden())
        feature = self.hidden2label(lstm_out[-1])
        
        if returns_feature:
            return feature
        else:
            return self.classifier(feature)

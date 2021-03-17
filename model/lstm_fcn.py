import torch


class lstm_fcn(torch.nn.Module):
    def __init__(self, in_channel=1, channel_1=128, channel_2=256, channel_3=128, kernel_size_1=8, kernel_size_2=5, kernel_size_3=3,
                 hidden_dim=32, num_classes=10):
        super(lstm_fcn, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_dim)
        self.dropout = torch.nn.Dropout(0.8)
        
        self.conv1 = torch.nn.Conv1d(in_channels=in_channel, out_channels=channel_1, kernel_size=kernel_size_1,
                                     padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm1d(channel_1)
        self.conv2 = torch.nn.Conv1d(in_channels=channel_1, out_channels=channel_2, kernel_size=kernel_size_2,
                                     padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm1d(channel_2)
        self.conv3 = torch.nn.Conv1d(in_channels=channel_2, out_channels=channel_3, kernel_size=kernel_size_3,
                                     padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm1d(channel_3)

        self.feature_dim = channel_3 + hidden_dim
        self.classifier = torch.nn.Linear(channel_3 + hidden_dim, num_classes)
    
    def forward(self, inp, returns_feature=False):
        # print('unsqueeze', input.shape)
        x = inp.permute(2, 0, 1)
        # print('permute', x.shape)
        _, (x, cell) = self.lstm(x)
        x = self.dropout(x)
        
        y = self.conv1(inp)
        y = self.bn1(y)
        y = torch.nn.functional.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = torch.nn.functional.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.avg_pool1d(y, kernel_size=y.shape[2])
        
        # print('x', x.shape)
        # print('y', y.shape)
        feature = torch.cat((x.squeeze(), y.squeeze()), dim=1)
        # print('cat', feature.shape)
        if returns_feature:
            return feature
        else:
            return self.classifier(feature)


def LSTM_FCN(input_size, hidden_dim, num_classes):
    return lstm_fcn(hidden_dim=hidden_dim, num_classes=num_classes)

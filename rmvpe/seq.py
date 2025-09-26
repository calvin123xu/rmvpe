import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.contiguous()  # 确保tensor连续
        return self.gru(x)[0]


class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.contiguous()  # 确保tensor连续
        return self.lstm(x)[0]


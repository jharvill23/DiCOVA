import torch.nn as nn
import torch.nn.functional as F
#import torch.autograd.function as Function
#from torch.autograd import function as Function
import torch
from easydict import EasyDict as edict
import yaml


CONFIG = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

class CTCmodel(nn.Module):
    def __init__(self, config):
        super(CTCmodel, self).__init__()
        self.config = config
        model = eval(self.config.model.name)
        self.model = model(config)

    def forward(self, padded_input):
        return self.model(padded_input)


class Fixed_length_autoencoder(nn.Module):
    def __init__(self, config):
        super(Fixed_length_autoencoder, self).__init__()
        self.config = config
        self.class_size = 30  # max possible intervals to count
        self.num_mels = config.data.num_mels + 1  # the extra dimension encodes the length of the audio in frames
        self.hidden_size = config.model.hidden_size
        self.bottleneck_size = config.model.bottleneck_size
        self.encoder_num_layers = config.model.encoder_num_layers
        self.decoder_num_layers = config.model.decoder_num_layers
        self.batch_first = config.model.batch_first
        self.dropout = config.model.dropout
        self.bidirectional = config.model.bidirectional
        self.encoder_lstm_1 = nn.LSTM(input_size=self.num_mels, hidden_size=self.hidden_size,
                             num_layers=self.encoder_num_layers-1, batch_first=self.batch_first,
                             dropout=self.dropout, bidirectional=self.bidirectional)
        self.encoder_lstm_2 = nn.LSTM(input_size=self.hidden_size if not self.bidirectional else self.hidden_size*2,
                                      hidden_size=self.bottleneck_size, num_layers=1, batch_first=self.batch_first,
                             dropout=self.dropout, bidirectional=self.bidirectional)
        self.decoder_lstm = nn.LSTM(input_size=self.bottleneck_size if not self.bidirectional else 2*self.bottleneck_size,
                                    hidden_size=self.hidden_size, num_layers=self.decoder_num_layers,
                                    batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        self.full1 = nn.Linear(in_features=self.hidden_size if not self.bidirectional else self.hidden_size*2,
                               out_features=300)
        self.full2 = nn.Linear(in_features=300, out_features=self.num_mels)

    def forward(self, x):
        input_dim = x.shape[1]  # check this!!!
        x, _ = self.encoder_lstm_1(x)
        x, _ = self.encoder_lstm_2(x)
        """Now we want the forward and backward summaries"""
        out_forward = x[:, :, :self.bottleneck_size]
        out_backward = x[:, :, self.bottleneck_size:]
        x_forward = out_forward[:, -1]
        x_backward = out_backward[:, 0]
        bottleneck_vector_ = torch.cat((x_forward, x_backward), dim=1)
        bottleneck_vector = torch.unsqueeze(bottleneck_vector_, dim=1)
        """Now we want to copy the bottleneck vector"""
        x = bottleneck_vector.repeat(1, input_dim, 1)
        x, _ = self.decoder_lstm(x)
        x = self.full1(x)
        x = F.tanh(x)
        x = self.full2(x)
        return x, bottleneck_vector_
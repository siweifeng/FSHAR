"""Models for HAR"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

        
class FFLSTMEncoder1(nn.Module):
    """FFLSTM encoder model for ADDA."""

    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, 
                 fc2_size):
        """Init FFLSTM encoder."""
        super(FFLSTMEncoder1, self).__init__()

        self.restored = False
        
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.fc2_size = fc2_size
        
        self.lstm = nn.LSTM(lstm_input_size, 
                            lstm_hidden_size, 
                            lstm_num_layers, 
                            batch_first=True)

        self.fc2 = nn.Linear(lstm_hidden_size, fc2_size) 

    def forward(self, x):
        """Forward the FFLSTM."""
        h0 = Variable(torch.zeros(self.lstm_num_layers,
                                  x.size(0),
                                  self.lstm_hidden_size)) 
        c0 = Variable(torch.zeros(self.lstm_num_layers,
                                  x.size(0),
                                  self.lstm_hidden_size))
        r_out, (h_n, h_c) = self.lstm(x, (h0, c0))   
        out2 = self.fc2(r_out[:, -1, :])
        return out2


class FFLSTMClassifier(nn.Module):
    """FFLSTM classifier model for ADDA."""

    def __init__(self, fc2_size, num_classes):
        """Init FFLSTM encoder."""
        super(FFLSTMClassifier, self).__init__()
        self.fc = nn.Linear(fc2_size, num_classes)

    def forward(self, feat):
        """Forward the FFLSTM classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc(out)
        return out
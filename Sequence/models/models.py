import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, node_type):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.node_type = node_type
        
        if node_type=='rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        elif node_type=='lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        elif node_type=='gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers)
            
    def forward(self, x):
        # gather weights in a contiguous memory location for 
        # more efficient processing
        self.rnn.flatten_parameters()
        
        h0 = torch.zeros(self.num_layers, x.size(1), 
                        self.hidden_size, device=x.device)
        
        if self.node_type == 'lstm':
            c0 = torch.zeros_like(h0)
            _, hidden = self.rnn(x.view(x.shape[0], x.shape[1], 
                                    self.input_size), (h0,c0))
        else:
            _, hidden = self.rnn(x.view(x.shape[0], x.shape[1], 
                                    self.input_size), h0)
        
        return hidden
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, node_type):
        super(Decoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.node_type = node_type
        
        if flavor=='rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        elif flavor=='lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        elif flavor=='gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers)
            
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, encoder_hidden):
        # gather weights in a contiguous memory location for 
        # more efficient processing
        self.rnn.flatten_parameters()
        
        out, hidden = self.rnn(x.unsqueeze(0), encoder_hidden)
        
        out = self.linear(out.squeeze(0))
        
        return out, hidden
    
class EncDec
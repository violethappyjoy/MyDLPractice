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
        
        if node_type=='rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        elif node_type=='lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        elif node_type=='gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers)
            
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, encoder_hidden):
        # gather weights in a contiguous memory location for 
        # more efficient processing
        self.rnn.flatten_parameters()
        
        out, hidden = self.rnn(x.unsqueeze(0), encoder_hidden)
        
        out = self.linear(out.squeeze(0))
        
        return out, hidden
    
class EncDec(nn.Module):
    '''
        Encoder Decoder model for Many-to-Many RNN
    '''
    def __init__(self, encoder, decoder, npred):
        '''
        Arguments
        encoder -- RNN for encoder
        decoder -- RNN for decoder
        npred -- Number of points to predict
        '''
        super(EncDec, self).__init__()
        
        self.enc   = encoder
        self.dec   = decoder
        self.npred = npred
        
    def forward(self, x):
        '''
        Arguments
        x -- input of shape sequence, batch_siz
        '''
        local_batch_size = x.shape[1]
        target_len = self.npred
        
        input_batch = x.unsqueeze(2)
        
        outs = torch.zeros(target_len, local_batch_size, 
                           input_batch.shape[2], device=x.device)
        
        enc_hid = self.enc(input_batch)
        
        dec_in = input_batch[-1, :, :]
        
        dec_hid = enc_hid
        
        # make prediction like a traditional RNN point-by-point
        #         by using the predicted point as new input
        for t in range(target_len):
            # note that the dec_hid is being continuously rewritten
            dec_out, dec_hid = self.dec(dec_in, dec_hid)
            # store the prediction
            outs[t] = dec_out
            # feed back the prediction as input to the decoder
            dec_in =  dec_out
            
        return outs.reshape(target_len, local_batch_size)
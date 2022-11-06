import torch
from torch import nn

train_on_GPU = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, 
                 input_size, # 37
                 hidden_size, # 64
                 num_layers):
        super(Encoder, self).__init__()
                
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True) # (batch_size, sequence_length, hidden_size)
        
    def forward(self, x, prev_state):
        _, state = self.lstm(x, prev_state)
        return state # (hidden_state, cell_state)
    

class Decoder(nn.Module):
    def __init__(self, 
                 input_size, # 11
                 hidden_size, # 64
                 output_size, # 11
                 num_layers):
        super(Decoder, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True) # (batch_size, sequence_length, hidden_size)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, prev_state):
        out, state = self.lstm(x, prev_state)
        yhat = self.fc(out)
        return yhat, state
    
    
class SeqtoSeq(nn.Module):
    def __init__(self, 
                 input_size_encoder, 
                 input_size_decoder,
                 hidden_size, 
                 output_size, 
                 num_layers,
                 max_machine_len):
        super(SeqtoSeq, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size 
        self.max_machine_len = max_machine_len
        
        self.encoder = Encoder(input_size_encoder, 
                  hidden_size, 
                  num_layers)
        self.decoder = Decoder(input_size_decoder, 
                  hidden_size, 
                  output_size,
                  num_layers)
        
    def forward(self, en_x, de_x, train):
        init_state = self.init_state(batch_size=en_x.shape[0])
        state = self.encoder(en_x, init_state)
        
        if train == True:
            # de_x: (batch_size, time_step (seq), vocab_size (feature))
            yhat, _ = self.decoder(de_x, state)
        elif train == False:
            yhat = []
            for i in range(self.max_machine_len):
                # de_x: (batch_size, 1, vocab_size (feature))
                de_x, state = self.decoder(de_x, state)
                yhat.append(de_x)
				# 
                _, idxs = torch.max(de_x, 2)
                de_x = nn.functional.one_hot(idxs, num_classes=11).float()
                #
            yhat = torch.cat(yhat, 1)
        return yhat
    
    def init_state(self, batch_size=1):
        if train_on_GPU:
            ht = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
            ct = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
        else:
            ht = torch.zeros((self.num_layers, batch_size, self.hidden_size))
            ct = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        return (ht, ct)
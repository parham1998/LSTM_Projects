import torch
from torch import nn

train_on_GPU = torch.cuda.is_available()

class LSTM(nn.Module):
    def __init__(self, 
                 input_size, # 38
                 hidden_size, # 64
                 output_size, # 38
                 num_layers=1, 
                 embedding=False,
                 emb_dimention=8):
        super(LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size    
        self.embedding = embedding
        
        if self.embedding == True:
            # Embedding needs words indices, not the one-hot vector
            self.word_embedding = nn.Embedding(input_size, emb_dimention)
            input_dimention = emb_dimention
        elif self.embedding == False:
            input_dimention = input_size
        
        self.lstm = nn.LSTM(input_size=input_dimention,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True) # (batch_size, sequence_length, hidden_size)
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x, prev_state):
        if self.embedding == True:
            x = self.word_embedding(x.long())
        # state = hidden state & cell state
        out, state = self.lstm(x, prev_state)
        out = self.dropout(out)
        yhat = self.fc(out)
        return yhat, state
        
    
    def init_state(self, batch_size=1):
        if train_on_GPU:
            ht = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
            ct = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
        else:
            ht = torch.zeros((self.num_layers, batch_size, self.hidden_size))
            ct = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        return (ht, ct)
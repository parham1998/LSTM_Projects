import torch
from torch import nn

train_on_GPU = torch.cuda.is_available()

class LSTM(nn.Module):
    def __init__(self, 
                 input_size, # 400001
                 hidden_size, # 128
                 output_size, # 5
                 embedding, 
                 emb_pretrained_weight,
                 num_layers, # 2
                 dropout):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers
        self.embedding = embedding
        self.hidden_size = hidden_size   
        
        if self.embedding == True: # use glove word-embedding
            emb_dimention = 50
            self.word_embeds = nn.Embedding(input_size, emb_dimention)
            # load pre-trained embedding weight
            self.word_embeds.weight.data.copy_(emb_pretrained_weight)
        elif self.embedding == False: # use one-hot-vector
            emb_dimention = input_size
        
        self.lstm = nn.LSTM(input_size=emb_dimention,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True, # (batch_size, sequence_length, hidden_size)
                            dropout = dropout) 
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, prev_state):
        if self.embedding == True:
            x = self.word_embeds(x.long())
        elif self.embedding == False:
            x = self.one_hot_vector(x)
        out, _ = self.lstm(x, prev_state)
        # just get the last step output
        out = out[:, -1, :]
        yhat = self.fc(out)
        return yhat
    
    def one_hot_vector(self, x):
        ohv = torch.zeros((x.size(0), x.size(1), self.input_size)).cuda()
        for i, _ in enumerate(x):
            for j, c in enumerate(x[i]):
                ohv[i, j, c.item()] = 1
        return ohv
        
    def init_state(self, batch_size=1):
        if train_on_GPU:
            ht = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
            ct = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
        else:
            ht = torch.zeros((self.num_layers, batch_size, self.hidden_size))
            ct = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        return (ht, ct)
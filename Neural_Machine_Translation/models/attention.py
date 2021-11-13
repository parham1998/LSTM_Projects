import torch
from torch import nn

train_on_GPU = torch.cuda.is_available()

class PreAttention(nn.Module):
    def __init__(self, 
                 input_size, # 37
                 hidden_size, # 32
                 num_layers=1):
        super(PreAttention, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True) # (batch_size, sequence_length, hidden_size)
        
    def forward(self, x, prev_state):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x, prev_state)
        return out # (batch_size, sequence_length, 2 * hidden_size)
    

class OneStepAttention(nn.Module):
    def __init__(self, 
                 hidden_size_pre, # 32
                 hidden_size_post): # 64
        super(OneStepAttention, self).__init__()
        
        self.fc1 = nn.Linear(2 * hidden_size_pre + hidden_size_post, 10) # 128 * 10 
        self.fc2 = nn.Linear(10, 1) # 10 * 1
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,
                pre_attention_hidden_states, # (batch_size, sequence_length, 2 * pre_attention_hidden_size)
                post_attention_hidden_state): # (1, batch_size, post_attention_hidden_size)
        
        s = post_attention_hidden_state.transpose(1, 0) # (batch_size, 1, post_attention_hidden_size)
        s = s.repeat(1, pre_attention_hidden_states.size(1), 1) # (batch_size, sequence_length, post_attention_hidden_size)
        concat = torch.cat((pre_attention_hidden_states, s), -1) # (batch_size, sequence_length, 2 * pre_attention_hidden_size + post_attention_hidden_size)
        e = torch.tanh(self.fc1(concat))
        energies = torch.relu(self.fc2(e))
        alphas = self.softmax(energies) # (batch_size, sequence_length, 1)
        context = torch.bmm(alphas.transpose(2,1), pre_attention_hidden_states)
        return context, alphas # context: (batch_size, 1, pre_attention_hidden_size)
   
    
class AttentionModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size_pre,  
                 hidden_size_post, 
                 output_size, 
                 num_layers,
                 max_machine_len):
        super(AttentionModel, self).__init__()
        
        self.hidden_size_pre = hidden_size_pre
        self.hidden_size_post = hidden_size_post
        self.max_machine_len = max_machine_len
        self.num_layers = num_layers
        
        self.preattention = PreAttention(input_size, 
                                         hidden_size_pre, 
                                         num_layers)
        
        self.onestepattention = OneStepAttention(hidden_size_pre,
                                                 hidden_size_post)
        
        self.postattention = nn.LSTM(input_size=2 * hidden_size_pre, # 64
                                     hidden_size=hidden_size_post, # 64
                                     num_layers=num_layers,
                                     batch_first=True) # (batch_size, sequence_length, hidden_size)
       
        self.fc = nn.Linear(hidden_size_post, output_size)
    
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        init_state_pre = self.init_state(D=2, batch_size=x.shape[0], hidden_size=self.hidden_size_pre)
        pre_attention_hidden_states = self.preattention(x, init_state_pre)
        
        init_state_post = self.init_state(D=1, batch_size=x.shape[0], hidden_size=self.hidden_size_post)
        post_attention_hidden_state = init_state_post
        yhat = []
        attention = []
        for i in range(self.max_machine_len):
            
            context, alphas = self.onestepattention(pre_attention_hidden_states, post_attention_hidden_state[0])
            
            out, post_attention_hidden_state = self.postattention(context, post_attention_hidden_state)
            out = self.fc(out)
            
            attention.append(alphas)
            yhat.append(out)
            
        yhat = torch.cat(yhat, 1)
        return yhat, attention
    
    def init_state(self, D=1, batch_size=1, hidden_size=32):
        if train_on_GPU:
            # D: bidirectional
            ht = torch.zeros((self.num_layers * D, batch_size, hidden_size)).cuda()
            ct = torch.zeros((self.num_layers * D, batch_size, hidden_size)).cuda()
        else:
            ht = torch.zeros((self.num_layers * D, batch_size, hidden_size))
            ct = torch.zeros((self.num_layers * D, batch_size, hidden_size))
        return (ht, ct)
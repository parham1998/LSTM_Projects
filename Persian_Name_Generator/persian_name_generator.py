# =============================================================================
# Import required libraries 
# =============================================================================
import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader

from lstm import LSTM

# =============================================================================
# Check if CUDA is available
# =============================================================================
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')
    print(torch.cuda.get_device_properties('cuda'))

# =============================================================================
# Load data & Clean data 
# =============================================================================
data = pd.read_csv('./dataset/دیتاست-اسامی-نام-های-فارسی.csv')
data.head()

persian_alphabet = ['آ', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع',
 'غ', 'ف', 'ق', 'ك', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ي', 'ئ', 'ء', 'ا', 'أ', ' ']

def clean(name):
    # remove spaces from the beginning and the end of name
    name = name.strip()
    # check if name only contains persian_alphabet
    name = "".join([ch for ch in name if ch in persian_alphabet])
    # add . to specify the end of the name
    name += "."
    return name
data_plus_dot = data['first_name'].apply(clean)

# build vocabulary
char_vocab = ["."] + [ch for ch in persian_alphabet]

# convert characters to integers and reverse
char_to_ix = {ch:i for i,ch in enumerate(char_vocab)}
ix_to_char = {i:ch for ch,i in char_to_ix.items()}

# =============================================================================
# Define dataset
# =============================================================================
class PersianNameDataset(Dataset):
    def __init__(self, data_plus_dot, max_namelen):        
        self.data_plus_dot = data_plus_dot 
        self.max_namelen = max_namelen

    def __getitem__(self, ix):
        # get data sample at index ix and change size to max_seqlen
        x_str = self.data_plus_dot[ix].ljust(self.max_namelen, ".")[:self.max_namelen]
        y_str = x_str[1:] + "."
        
        x_ohv = torch.zeros((self.max_namelen, len(char_vocab)))
        x = torch.zeros(self.max_namelen)
        y = torch.zeros(self.max_namelen)
        # change input data to one-hot vectors
        for i, c in enumerate(x_str):
            x_ohv[i, char_to_ix[c]] = 1
            x[i] = char_to_ix[c]            
        for i, c in enumerate(y_str):
            y[i] = char_to_ix[c]
        return x_ohv, x, y
    
    def __len__(self):
        return len(self.data_plus_dot)

max_namelen = 10
dataset = PersianNameDataset(data_plus_dot, max_namelen=max_namelen)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
one_batch = iter(dataloader).next()
print(one_batch[0].size(), one_batch[1].size(), one_batch[2].size())

# =============================================================================
# LSTM model
# =============================================================================
input_size = len(char_vocab)
hidden_size = 64
output_size = len(char_vocab)
num_layers = 1
embedding = True
emb_dimention = 8

if embedding == True:
    PATH = './checkpoints/LSTM_embedding_persian_name.pth'
elif embedding == False:
    PATH = './checkpoints/LSTM_persian_name.pth'

model = LSTM(input_size, 
             hidden_size, 
             output_size, 
             num_layers,  
             embedding,
             emb_dimention)
if train_on_GPU:
    model.cuda()
    print('\n model can be trained on gpu') 

# =============================================================================
# Load model
# =============================================================================
if train_on_GPU:
    model.load_state_dict(torch.load(PATH))
    model.cuda()
    print('\n net can be trained on gpu') 
else:
    model.load_state_dict(torch.load(PATH, torch.device('cpu')))
    
# =============================================================================
# Specify loss function and optimizer
# =============================================================================   
epochs = 50

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)    
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=len(dataloader), gamma=0.95)

def count_learnable_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_learnable_parameters(model)

# =============================================================================
# Training
# =============================================================================
epoch_losses = []
epoch_lrs = []
    
def train(epoch, clip=False):
    
    epoch_loss = 0
    epoch_lr = 0
    
    model.train()
    for batch_idx, (X_ohv, X, Y) in enumerate(dataloader):
        
        if train_on_GPU:
            X_ohv, X, Y = X_ohv.cuda(), X.cuda(), Y.cuda()
        
        # initialise model's state and perform forward-prop
        prev_state = model.init_state(batch_size=X.shape[0])
        if embedding == True:
            out, state = model(X, prev_state)
        elif embedding == False:
            out, state = model(X_ohv, prev_state)
        out = out.transpose(1, 2)
        
        optimizer.zero_grad()
            
        loss = criterion(out, Y.long())
            
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step() 
        lr_scheduler.step()
        
        epoch_loss += loss.item()
        epoch_lr += lr_scheduler.get_lr()[0]
        
    epoch_losses.append(epoch_loss/(batch_idx+1))
    epoch_lrs.append(epoch_lr/(batch_idx+1))
    print('Epoch: {} \t Training Loss: {:.3f}'.format(epoch+1, epoch_loss/(batch_idx+1)))
    torch.save(model.state_dict(), PATH)
    
print('==> Start Training ...')
for epoch in range(epochs):
    start = timeit.default_timer()
    train(epoch)
    stop = timeit.default_timer()
    print('time: {:.3f}'.format(stop - start))
print('==> End of training ...')

# =============================================================================
# Figure loss & lr
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
ax1.plot(epoch_losses, marker="o", markersize=5)
ax1.set_title("Loss")
ax2.plot(epoch_lrs, marker="o", markersize=5)
ax2.set_title("LR")
plt.xlabel("Epochs")
plt.show()

# =============================================================================
# Define sampler
# =============================================================================
def sampler(model, start='پ', k=5):
    
    if len(start) >= max_namelen:
        return start
    
    model.eval()
    with torch.no_grad():
        
        state = model.init_state(batch_size=1)
        length = 0
        name = start
        
        for char in start:
            if embedding == True:
                # (batch size, timestep)
                X = torch.zeros((1, 1)) 
                X[0, 0] = char_to_ix[char]
            elif embedding == False:
                # (batch size, timestep, vocabulary size)
                X = torch.zeros((1, 1, len(char_vocab))) 
                X[0, 0, char_to_ix[char]] = 1
            if train_on_GPU:
                X = X.cuda()
            out, state = model(X, state)
            length += 1
            
        vals, idxs = torch.topk(out[0], k) 
        idx = np.random.choice(idxs.cpu().numpy()[0])
        char = ix_to_char[idx]
        name += char
        
        while char != "." and length <= max_namelen-1:
            if embedding == True:
                # (batch size, timestep)
                X = torch.zeros((1, 1)) 
                X[0, 0] = char_to_ix[char]
            elif embedding == False:
                # (batch size, timestep, vocabulary size)
                X = torch.zeros((1, 1, len(char_vocab))) 
                X[0, 0, char_to_ix[char]] = 1
            if train_on_GPU:
                X = X.cuda()
            out, state = model(X, state)
            
            vals, idxs = torch.topk(out[0], k)
            idx = np.random.choice(idxs.cpu().numpy()[0])
            char = ix_to_char[idx]
            
            length += 1
            name += char
    
        if name[-1] != ".":
            name += "."
    
    return name

sampler(model, start='محم', k=3)
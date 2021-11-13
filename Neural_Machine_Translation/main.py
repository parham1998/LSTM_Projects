# =============================================================================
# Import required libraries 
# =============================================================================
import timeit

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader

from models import *
from generate_data import load_dataset

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
# Generate data
# =============================================================================
m = 11000
data, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

print(data[:10])
print(human_vocab)
print(machine_vocab)
print(inv_machine_vocab)

# =============================================================================
# Define dataset in pytorch format
# =============================================================================
def string_to_int(string, length, vocab):
        string = string.lower()
        string = string.replace(',', '')
    
        if len(string) > length:
            string = string[:length]
        rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
        if len(string) < length:
            rep += [vocab['<pad>']] * (length - len(string))
        return rep

class DateDataset(Dataset):
    def __init__(self, data, max_human_len, max_machine_len, human_vocab, machine_vocab):        
        self.data = data 
        self.max_human_len = max_human_len
        self.max_machine_len = max_machine_len
        self.human_vocab = human_vocab
        self.machine_vocab = machine_vocab
                
    def __getitem__(self, ix):
        x = torch.zeros(self.max_human_len, len(self.human_vocab))
        # y_ohv used as input to decoder
        y_ohv = torch.zeros(self.max_machine_len, len(self.machine_vocab))
        # y used as output
        y = torch.zeros(self.max_machine_len)
            
        # change input data to one-hot vectors
        rep_x = string_to_int(self.data[ix][0], self.max_human_len, self.human_vocab)
        for i in range(self.max_human_len):
            x[i, rep_x[i]] = 1
        rep_y = string_to_int(self.data[ix][1], self.max_machine_len, self.machine_vocab)
        for i in range(self.max_machine_len):
            y_ohv[i, rep_y[i]] = 1
        for i in range(self.max_machine_len):
            y[i] = rep_y[i]
            
        return x, y_ohv, y
    
    def __len__(self):
        return len(self.data)

max_human_len = 30
max_machine_len = 10
train_data = DateDataset(data[:10000], max_human_len, max_machine_len, human_vocab, machine_vocab)
test_data = DateDataset(data[10000:], max_human_len, max_machine_len, human_vocab, machine_vocab)

batch_size = 64
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

one_batch = iter(trainloader).next()
print(one_batch[0].size(), one_batch[1].size(), one_batch[2].size())

# =============================================================================
# Models
# =============================================================================
attention = True
if attention == True:
    # attention model
    input_size = len(human_vocab)
    hidden_size_pre = 32
    hidden_size_post = 64
    output_size = len(machine_vocab)
    num_layers = 1

    PATH = './checkpoints/AttentionModel.pth'
    model = AttentionModel(input_size, 
                           hidden_size_pre, 
                           hidden_size_post,
                           output_size, 
                           num_layers,
                           max_machine_len)
    if train_on_GPU:
        model.cuda()
        print('\n model can be trained on gpu') 
        
elif attention == False:
    # seq to seq model
    input_size = len(human_vocab)
    hidden_size = 64
    output_size = len(machine_vocab)
    num_layers = 1

    PATH = './checkpoints/SeqtoSeq.pth'
    model = SeqtoSeq(input_size,
                     output_size,
                     hidden_size, 
                     output_size,
                     num_layers,
                     max_machine_len)
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
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)    
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.8)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters())
count_parameters(model)

# =============================================================================
# Training
# =============================================================================
best_accuracy = 0

epoch_losses = []
epoch_accuracy = []
epoch_lrs = []

# for seq to seq model
def zero_concatination(x):
    if train_on_GPU:
        zero = torch.zeros([x.shape[0], 1, x.shape[2]]).cuda()
    else:
        zero = torch.zeros([x.shape[0], 1, x.shape[2]])
    return torch.cat((zero, x[:,:-1,:]), 1)

def train(epoch, clip=False):
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.train()
    for batch_idx, (X, Y_inp, Y) in enumerate(trainloader):
        
        if train_on_GPU:
            X, Y_inp, Y = X.cuda(), Y_inp.cuda(), Y.cuda()

        if attention == False:      
            out = model(X, zero_concatination(Y_inp), True)
        elif attention == True:
            out, _ = model(X)
        out = out.transpose(1, 2)
        
        optimizer.zero_grad()
            
        loss = criterion(out, Y.long())
            
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step() 
        
        epoch_loss += loss.item()
        _, predicted = torch.max(out, 1)
        total += Y.size(0)
        correct += (predicted == Y).sum().item()
    
    print('Epoch: {} \t Training Loss: {:.3f} \t Training Accuracy: {:.3f}'.format(epoch+1, epoch_loss/(batch_idx+1), 10.*correct/total))

def test(epoch):
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, Y_inp, Y) in enumerate(testloader):
        
            if train_on_GPU:
                X, Y_inp, Y = X.cuda(), Y_inp.cuda(), Y.cuda()
        
            if attention == False:       
                out = model(X, torch.zeros([Y_inp.shape[0], 1, Y_inp.shape[2]]).cuda(), False)
            elif attention == True: 
                out, _ = model(X)
            out = out.transpose(1, 2)
            
            loss = criterion(out, Y.long())
             
            epoch_loss += loss.item()
            _, predicted = torch.max(out, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
    
    acc = 10.*correct/total
    epoch_losses.append(epoch_loss/(batch_idx+1))
    epoch_accuracy.append(acc)
    print('Epoch: {} \t Test Loss: {:.3f} \t Test Accuracy: {:.3f}'.format(epoch+1, epoch_loss/(batch_idx+1), acc))
    
    # save model if test accuracy has increased 
    global best_accuracy
    if acc > best_accuracy:
        print('Test accuracy increased ({:.3f} --> {:.3f}). saving model ...'.format(best_accuracy, acc))
        torch.save(model.state_dict(), PATH)
        best_accuracy = acc
    
print('==> Start Training ...')
for epoch in range(epochs):
    start = timeit.default_timer()
    train(epoch)
    test(epoch)
    lr_scheduler.step()
    epoch_lrs.append(lr_scheduler.get_lr()[0])
    stop = timeit.default_timer()
    print('time: {:.3f}'.format(stop - start))
print('==> End of training ...')

# =============================================================================
# Figure accuracy, loss & lr
# =============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 12))
ax1.plot(epoch_accuracy, marker="o", markersize=5)
ax1.set_title("accuracy")
ax2.plot(epoch_losses, marker="o", markersize=5)
ax2.set_title("Loss")
ax3.plot(epoch_lrs, marker="o", markersize=5)
ax3.set_title("LR")
plt.xlabel("Epochs")
plt.show()

# =============================================================================
# Test your date
# =============================================================================
def sample(date):
    x = torch.zeros(max_human_len, len(human_vocab))
    rep_x = string_to_int(date, max_human_len, human_vocab)
    for i in range(max_human_len):
        x[i, rep_x[i]] = 1
    
    if train_on_GPU:
        x = x.unsqueeze(0).cuda()
        
    if attention == False:      
        out = model(x, torch.zeros([1, 1, len(machine_vocab)]).cuda(), False)
    elif attention == True: 
        out, attentions = model(x)
    out = out.transpose(1, 2)
        
    _, predicted = torch.max(out, 1)
    p = ''
    for i in range(max_machine_len):
        p += inv_machine_vocab[predicted[0][i].item()]
    
    if attention == False:  
        return p
    elif attention == True: 
        return p, attentions
    
human_date = '10.11.19'
if attention == False:  
    machine_date = sample(human_date)
if attention == True: 
    machine_date, attentions = sample(human_date)
print("human date is: {}, machine date is: {}".format(human_date, machine_date))

# =============================================================================
# visualization of attention weights
# =============================================================================
def attention_weights(human_date, machine_date, attentions):
    attentions_np = np.zeros(shape=(max_machine_len, max_human_len))
    for i in range(max_machine_len):
        for j in range(max_human_len):
            attentions_np[i][j] = attentions[i][0,j,0].item()

    fig, ax = plt.subplots(figsize=(10, 30))
    im = ax.imshow(attentions_np)
    ax.set_xticks(np.arange(max_human_len))
    ax.set_yticks(np.arange(max_machine_len))

    rep = string_to_int(human_date, max_human_len, human_vocab)
    inv_human_vocab = {i:ch for ch,i in human_vocab.items()}
    human_date_list = []
    for i in range(max_human_len):
        human_date_list.append(inv_human_vocab[rep[i]])

    ax.set_xticklabels(human_date_list)
    ax.set_yticklabels(list(machine_date))              
    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")                

    fig.tight_layout()
    plt.show()
attention_weights(human_date, machine_date, attentions)
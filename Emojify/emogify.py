# =============================================================================
# Import required libraries 
# =============================================================================
# pip install emoji
import emoji

import timeit
import csv

import numpy as np
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
# Glove word-embedding 
# =============================================================================
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

# download "glove.6B.50d.txt" and place it in glove folder
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./glove/glove.6B.50d.txt')
print(index_to_word[336275])
print(word_to_index['soccer'])
print(word_to_vec_map['soccer'])

# =============================================================================
# Show emojies (❤️, ⚾, 😄, 😞, 🍴)
# =============================================================================
emoji_dictionary = {"0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

label_to_emoji(1)

# =============================================================================
# Define dataset
# =============================================================================
def sentences_to_indices(X, word_to_index, max_seqlen):
    m = X.shape[0] # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_seqlen))
    
    for i in range(m):
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        
        j = 0
        
        for w in sentence_words:
            # if w exists in the word_to_index dictionary
            if w in word_to_index:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j = j + 1
            if j == max_seqlen:
                break
    return X_indices

print(sentences_to_indices(np.array(['Goey does not share food !']), word_to_index, 8))

class EmojiDataset(Dataset):
    def __init__(self, root, max_seqlen):        
        self.root = root 
        
        sentences = []
        labels = []
        
        with open (root) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            next(csvReader)
            for row in csvReader:
                sentences.append(row[0])
                labels.append(row[1])

            sentences = np.asarray(sentences)
            labels = np.asarray(labels, dtype=int)  
        
        self.sentences_to_indices = sentences_to_indices(sentences, word_to_index, max_seqlen)
        self.labels = labels
        
    def __getitem__(self, ix):
        sentence = self.sentences_to_indices[ix]
        label = self.labels[ix]            
        return torch.tensor(sentence), torch.tensor(label)
    
    def __len__(self):
        return len(self.sentences_to_indices)

max_seqlen = 8
train_data = EmojiDataset('./dataset/train_emoji.csv', max_seqlen=max_seqlen)
test_data = EmojiDataset('./dataset/test_emoji.csv', max_seqlen=max_seqlen)

batch_size = 16
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

one_batch = iter(trainloader).next()
print(one_batch[0].size(), one_batch[1].size())

# =============================================================================
# LSTM model
# =============================================================================
vocab_size = len(word_to_index) + 1

def pretrained_embedding_layer(vocab_size, word_to_vec_map, word_to_index):
    emb_matrix = np.zeros((vocab_size, 50))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    return torch.tensor(emb_matrix) 
pretrained_weight = pretrained_embedding_layer(vocab_size, word_to_vec_map, word_to_index)

# use glove embedding with dimention 50 or use one-hot-vector with dimention 400001
embedding = True
if embedding == True:
    PATH = './checkpoints/LSTM_emogify_glove_.pth'
elif embedding == False:
    PATH = './checkpoints/LSTM_emogify_one_hot_vector.pth'
model = LSTM(input_size = vocab_size, 
             hidden_size = 128, 
             output_size = 5, 
             embedding = embedding,
             emb_pretrained_weight = pretrained_weight,
             num_layers = 2, 
             dropout = 0.5)
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

if embedding == True:
    for param in model.word_embeds.parameters():
        param.requires_grad = False

params = [p for p in model.parameters() if p.requires_grad == True]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params, lr=1e-3)    
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.8)

def count_all_parameters(model): 
    return sum(p.numel() for p in model.parameters())
def count_learnabel_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_all_parameters(model)
count_learnabel_parameters(model)

# =============================================================================
# Training
# =============================================================================
best_accuracy = 0

epoch_losses = []
epoch_accuracy = []
epoch_lrs = []
    
def train(epoch, clip=False):
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.train()
    for batch_idx, (X, Y) in enumerate(trainloader):
        
        if train_on_GPU:
            X, Y = X.cuda(), Y.cuda()
        
        # initialise model's state and perform forward-prop
        prev_state = model.init_state(batch_size=X.shape[0])
        out = model(X, prev_state)
        
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
    
    print('Epoch: {} \t Training Loss: {:.3f} \t Training Accuracy: {:.3f}'.format(epoch+1, epoch_loss/(batch_idx+1), 100.*correct/total))

def test(epoch):
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(testloader):
        
            if train_on_GPU:
                X, Y = X.cuda(), Y.cuda()
        
            # initialise model's state and perform forward-prop
            prev_state = model.init_state(batch_size=X.shape[0])
            out = model(X, prev_state)
        
            loss = criterion(out, Y.long())
             
            epoch_loss += loss.item()
            _, predicted = torch.max(out, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
    
    acc = 100.*correct/total
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
# Test your sentence
# =============================================================================
def generate_emoji(sentence):
    sen = sentences_to_indices(np.array([sentence]), word_to_index, max_seqlen)
    if train_on_GPU:
        sen = torch.Tensor(sen).cuda()
    first_state = model.init_state(batch_size=1)
    out = model(sen, first_state)
    _, predicted = torch.max(out, 1)
    print("Sentence is: {}, Emoji is: {}".format(sentence, label_to_emoji(predicted.item())))

generate_emoji('teeny tiny girl')
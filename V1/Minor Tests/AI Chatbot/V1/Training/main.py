import numpy as np
import pandas as pd
import matplotlib as plt

import re
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, Dataset


# Define common contractions
CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "won't": "will not", "can't": "cannot", "don't": "do not"
}


class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length=512):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_size))

        # Create transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(embed_size, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        sequence_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :sequence_length, :]
        x = x.permute(sequence_length, batch_size, self.embed_size)
        x = self.transformer_encoder(x)
        x = x.permute(sequence_length, batch_size, self.embed_size)
        x = self.fc(x)
        return x


# Custom CrossEntropyLoss function for multiple ignore indices
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_indices):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ignore_indices = ignore_indices

    def forward(self, outputs, targets):
        # Flatten inputs and targets
        outputs = outputs.view(-1, outputs.size(-1))  # Shape: (N*seq_len, num_classes)
        targets = targets.view(-1)

        # Create a mask for the ignored indices
        mask = ~torch.tensor([target in self.ignore_indices for target in targets]).to(targets.device)

        # Apply the mask to the outputs and targets
        outputs = outputs[mask]
        targets = targets[mask]

        return nn.CrossEntropyLoss()(outputs, targets)


# function to expand contractions
def expand_contractions(text):
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(r'\b{}\b)'.format(contraction), expanded, text)

    return text


# Tokenization function
def tokenize(text):
    text = expand_contractions(text.lower())
    tokens = re.findall(r'\w+|\S', text)

    return tokens


# Byte Pair Encoding (BPE) for subword tokenization
class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bpe_codes = {}

    def get_vocab(self, tokenized_texts):
        pairs = defaultdict(int)

        for sentence in tokenized_texts:
            for word in sentence:
                symbols = list(word) + ['</w']

                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += 1

        return pairs

    def merge_vocab(self, pair, vocab):
        merged_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in vocab.items():
            merged_word = re.sub(r'\b{}\b'.format(bigram), replacement, ' '.join(word)).split()
            merged_vocab[tuple(merged_word)] = freq

        return merged_vocab

    def fit(self, tokenized_texts):
        vocab = Counter()

        for sentence in tokenized_texts:
            for word in sentence:
                vocab[tuple(list(word) + ['</w>'])] += 1

        for _ in range(self.vocab_size - len(vocab)):
            pairs = self.get_vocab(tokenized_texts)

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.bpe_codes[best_pair] = len(self.bpe_codes)

    def encode(self, text):
        tokens = list(text) + ['</w>']

        while True:

            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

            if not any(pair in self.bpe_codes for pair in pairs):
                break

            best_pair = min(pairs, key=lambda pair: self.bpe_codes.get(pair, float('inf')))
            first, second = best_pair
            new_token = first + second
            tokens = [new_token if i == first and j == second else i for i, j in zip(tokens, tokens[1:] + [None])]

        return tokens[:-1]


# Function to build the vocabulary
def build_vocab(tokenized_texts, vocab_size):
    bpe = BPE(vocab_size=vocab_size)
    bpe.fit(tokenized_texts)

    vocab = defaultdict(int)

    for sentence in tokenized_texts:
        for word in sentence:
            encoded_word = bpe.encode(word)
            vocab[tuple(encoded_word)] += 1

    # Create word-to-id and id-to-word
    word_to_id = {word: idx for idx, word in enumerate(vocab, start=2)}

    word_to_id['<pad>'] = 0
    word_to_id['<unk>'] = 1

    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return word_to_id, id_to_word


# Function to encode the dataset
def encode(text, word_to_id, max_length):

    # Tokenize and expand contractions
    tokens = tokenize(text)
    encoded_tokens = []

    for token in tokens:
        encoded_word = word_to_id.get(tuple(token), word_to_id['<unk>'])
        encoded_tokens.append(encoded_word)

    # Padding
    if len(encoded_tokens) > max_length:
        encoded_tokens = encoded_tokens[:max_length]
    else:
        encoded_tokens += [word_to_id['<pad>']] * (max_length - len(encoded_tokens))

    return encoded_tokens


# Function to train the model
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            # Get input tensor and labels
            input_ids = batch[0].to(device)
            labels = input_ids.clone().to(device)

            # Forward pass
            outputs = model(input_ids)

            # Compute loss
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}')

        # Evaluation Loop
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:

                # Get input tensor and labels
                input_ids = batch[0].to(device)
                labels = input_ids.clone().to(device)

                # Forward pass
                outputs = model(input_ids)

                # Compute loss
                loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
                val_loss += loss.item()

        print(f'Validation Loss: {val_loss / len(val_dataloader)}')


# Function to evaluate the model
def evaluate_model(model, test_dataloader, criterion, device):

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_dataloader)
    accuracy = total_correct / total_samples

    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    return avg_loss, accuracy




# Step 1: Check if CUDA is available
# If so, use GPU
# Else, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

print(torch.cuda.is_available())


# Step 2: Defining variable values

num_heads = 8      # Number of attention heads
num_layers = 6     # Number of transformer layers
num_epochs = 25    # Number of passes through the training dataset

batch_size = 8
embed_size = 128   # Size of embeddings

hidden_dim = 512   #
max_length = 512   # Maximum sequence length



# Step 3: Load the dataset
# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
val_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='val')

# print(dataset['text'][3])


# Step 4: Tokenize the dataset
# tokenized_dataset = [tokenize(sentence) for sentence in dataset]
tokenized_train_dataset = [tokenize(sentence) for sentence in train_dataset]
tokenized_test_dataset = [tokenize(sentence) for sentence in test_dataset]
tokenized_val_dataset = [tokenize(sentence) for sentence in val_dataset]


# Step 5: Find the number of unique tokens
unique_tokens = set()

for sentence in tokenized_train_dataset:
    unique_tokens.update(sentence)

vocab_size = len(unique_tokens)


# Step 6: Build the vocabulary
# word_to_id, id_to_word = build_vocab(tokenized_dataset, vocab_size=50000)
# vocab_size = len(word_to_id)

train_word_to_id, train_id_to_word = build_vocab(tokenized_train_dataset, vocab_size)


# Step 7: Encode the dataset
# encoded_dataset = [encode(sentence, word_to_id, max_length) for sentence in dataset]

train_encoded_dataset = [encode(sentence, train_word_to_id, max_length) for sentence in train_dataset]
test_encoded_dataset = [encode(sentence, train_word_to_id, max_length) for sentence in test_dataset]
val_encoded_dataset = [encode(sentence, train_word_to_id, max_length) for sentence in val_dataset]


# Step 8: Create data loaders
# train_dataset = encoded_dataset['train']
# test_dataset = tokenized_dataset['test']
# val_dataset = encoded_dataset['validation']

# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=8)
# val_dataloader = DataLoader(val_dataset, batch_size=8)

train_dataloader = DataLoader(train_encoded_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_encoded_dataset, batch_size=8)
val_dataloader = DataLoader(val_encoded_dataset, batch_size=8)


# Step 9: Define the model
model = TransformerLanguageModel(vocab_size, embed_size, num_heads, num_layers, max_length)
model = model.to(device)

model.train()


# Step 10: Create the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # learning rate

ignore_indices = [train_word_to_id['<pad>'], train_word_to_id['<unk>']]
criterion = CustomCrossEntropyLoss(ignore_index=ignore_indices)
# criterion = nn.CrossEntropyLoss(ignore_index=word_to_id['<unk>'])


# Step 11: Train the model
train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs)


# Step 12: Evaluate the model
model.eval()


# Step 13: Save the model
torch.save(model.state_dict(), 'transformer_language_model.pth')

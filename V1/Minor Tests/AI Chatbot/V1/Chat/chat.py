import torch
import torch.nn as nn
from collections import Counter
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long)


class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        for transformer in self.transformer_blocks:
            x = transformer(x)
        logits = self.fc_out(x)
        return logits


def build_vocabulary(dataset):
    counter = Counter()

    for item in dataset:
        counter.update(item['text'].split())
    vocab = {word: i for i, (word, _) in enumerate(counter.most_common())}
    vocab['<unk>'] = len(vocab)
    return vocab


def preprocess_dataset(dataset, word_to_id):
    input_ids = []

    for item in dataset:
        tokens = tokenize(item['text'], word_to_id)
        input_ids.append(tokens)

    return input_ids


def tokenize(text, word_to_id):
    return [word_to_id.get(word, word_to_id['<unk>']) for word in text.split()]


def decode(tokens, id_to_word):
    return ' '.join([id_to_word[token] for token in tokens if token in id_to_word])


def collate(batch):
    return pad_sequence(batch, batch_first=True, padding_value=word_to_id['<unk>'])


def generate_response(input_text, model, word_to_id, id_to_word, max_length=500):
    tokens = tokenize(input_text, word_to_id)
    #input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    input_tensor = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        output_tokens = model(input_tensor)

    generated_tokens = output_tokens[0, -1, :].topk(1)[1]
    generated_text = decode(generated_tokens.cpu().numpy(), id_to_word)

    return generated_text


#if torch.cuda.is_available():
#    device = torch.device("cuda")
#    print("Using GPU: ", torch.cuda.get_device_name(0))
#else:
#    device = torch.device("cpu")
#    print("Using CPU")

vocab_size = 50000
hidden_dim = 512
num_layers = 6

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

vocab = build_vocabulary(dataset)
vocab_size = len(vocab)

word_to_id = vocab
id_to_word = {id: word for word, id in vocab.items()}
tokenized_dataset = preprocess_dataset(dataset, word_to_id)
text_dataset = TextDataset(tokenized_dataset)

model = GPT(vocab_size, hidden_dim, num_layers)
model.load_state_dict(torch.load('gpt_model.pth'))
model.eval()

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    response = generate_response(user_input, model, word_to_id, id_to_word)
    print("AI: ", response)
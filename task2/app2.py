import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import math


def preprocess_text(text):
    text = text.lower().replace("\n", " ")
    text = re.sub(" +", " ", text)
    return ''.join(c for c in text if c.isalpha() or c.isspace())


class TextDataset(Dataset):
    def __init__(self, text, sequence_length, vocab):
        self.text = text
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.encoded_text = [self.char_to_idx[c] for c in text]
        self.inputs, self.targets = self._create_sequences()

    def _create_sequences(self):
        inputs, targets = [], []
        for i in range(len(self.encoded_text) - self.sequence_length):
            inputs.append(self.encoded_text[i:i + self.sequence_length])
            targets.append(self.encoded_text[i + self.sequence_length])
        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out[:, -1, :])
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead=8, num_layers=2, dim_feedforward=2048):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # Apply embedding and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)

        # Get predictions using the last sequence element
        logits = self.fc(transformer_out[:, -1, :])
        return logits

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        """
        Load model from checkpoint while ensuring dimension consistency
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # Extract dimensions from the checkpoint
        fc_weight = model_state['fc.weight']
        vocab_size, embedding_dim = fc_weight.shape

        # Create model with matching dimensions
        model = cls(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=8  # You can adjust these default values as needed
        )

        # Load the state dict
        model.load_state_dict(model_state)
        return model


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, name):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'./models/{name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


# Generate text
def generate_text(model, seed_text, length, vocab):
    model.eval()
    char_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_char = {idx: word for word, idx in char_to_idx.items()}

    generated_text = seed_text
    input_seq = torch.tensor([char_to_idx[char] for char in generated_text], dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(length):
        with torch.no_grad():
            output = model(input_seq)
            next_char_idx = temperature_sampling(output)
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)


    return ''.join(generated_text)


def temperature_sampling(probabilities: torch.Tensor, temperature=0.1) -> int:
    probabilities = torch.clamp(probabilities, min=1e-10)
    scaled_logits = torch.log(probabilities) / temperature
    adjusted_probs = torch.softmax(scaled_logits, dim=-1)
    sampled_idx = torch.multinomial(adjusted_probs, num_samples=1).item()

    return sampled_idx


# Main script
if __name__ == "__main__":
    with open("./data/pantadeusz.txt", "r", encoding="utf8") as f:
        text = preprocess_text(f.read())

    # Parameters
    seq_length = 50
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 4
    num_heads = 4
    batch_size = 32
    epochs = 50
    patience = 5
    test_length = 300

    # Create vocabulary
    vocab = sorted(set(text))
    vocab_size = len(vocab)

    # Split data
    tokens = text.split()
    train_end = int(0.8 * len(tokens))
    val_end = int(0.9 * len(tokens))

    train_text = ' '.join(tokens[:train_end])
    val_text = ' '.join(tokens[train_end:val_end])
    test_text = ' '.join(tokens[val_end:])

    train_dataset = TextDataset(train_text, seq_length, vocab)
    val_dataset = TextDataset(val_text, seq_length, vocab)
    test_dataset = TextDataset(test_text, seq_length, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = TransformerModel(vocab_size, embedding_dim).to(device)
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Training model")
    # train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, "transformer_mr_tadeusz")

    # Load best model
    model.load_state_dict(torch.load('./models/best_mr_tadeusz_medium_long.pth'), strict=False)

    # test
    seed_text = "gdziem rzadko płakał a nigdy nie zgrzytał kraje dzieciństwa"
    print("Generated text:", generate_text(model, seed_text, test_length, vocab), ";")


# Parameters - small
# seq_length = 15
# embedding_dim = 64
# hidden_dim = 128
# num_layers = 2
# batch_size = 32
# epochs = 50
# patience = 5
# test_length = 50

# Parameters - medium
# seq_length = 15
# embedding_dim = 128
# hidden_dim = 256
# num_layers = 3
# batch_size = 32
# epochs = 50
# patience = 5
# test_length = 50

# Parameters - medium long
# seq_length = 50
# embedding_dim = 128
# hidden_dim = 256
# num_layers = 4
# batch_size = 32
# epochs = 50
# patience = 5
# test_length = 100

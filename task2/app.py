import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from task2.MrTeddyLSTM import MrTeddyLSTM
from task2.data import preprocess_text, TextDataset
from task2.train import train_model
from task2.utils import generate_text, test_model

# Main script
if __name__ == "__main__":
    with open("./data/pantadeusz.txt", "r", encoding="utf8") as f:
        text = preprocess_text(f.read())

    # Parameters
    seq_length = 50
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 5
    batch_size = 32
    epochs = 50
    patience = 5
    test_length = 100
    temperature = 0.40

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MrTeddyTD(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, seq_length).to(device)
    model = MrTeddyLSTM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, "best_mr_tadeusz_dropout", device)

    # Load best model
    model.load_state_dict(torch.load('./models/best_mr_tadeusz_dropout.pth'), strict=False)

    # test
    seed_text = " bernardynem grał w mariasza i właśnie z wyświeconym winem"
    print("Generated text:", generate_text(model, seed_text, test_length, vocab, temperature, device), ";")

    test_model(model, test_loader, criterion, vocab, device)


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

# Parameters - transformer
# seq_length = 50
# embedding_dim = 256
# hidden_dim = 512
# num_layers = 4
# num_heads = 8
# batch_size = 32
# epochs = 50
# patience = 5
# test_length = 100
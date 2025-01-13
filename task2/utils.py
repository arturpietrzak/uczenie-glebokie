import torch


def generate_text(model, seed_text, length, vocab, temperature, device):
    model.eval()
    char_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_char = {idx: word for word, idx in char_to_idx.items()}

    generated_text = seed_text
    input_seq = torch.tensor([char_to_idx[char] for char in generated_text], dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(length):
        with torch.no_grad():
            output = model(input_seq)
            next_char_idx = temperature_sampling(output, temperature)
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char
            input_seq = torch.cat(
                [input_seq[:, 1:],
                 torch.tensor([[next_char_idx]], dtype=torch.long).to(device)],
                dim=1
            )

    return ''.join(generated_text)


def temperature_sampling(probabilities: torch.Tensor, temperature=0.001) -> int:
    probabilities = torch.clamp(probabilities, min=1e-10)
    scaled_logits = torch.log(probabilities) / temperature
    adjusted_probs = torch.softmax(scaled_logits, dim=-1)
    sampled_idx = torch.multinomial(adjusted_probs, num_samples=1).item()

    return sampled_idx


def test_model(model, test_loader, criterion, vocab, device):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Top-1 and Top-3 accuracy
            _, top1_pred = outputs.topk(1, dim=1)
            _, top3_pred = outputs.topk(3, dim=1)

            correct_top1 += (top1_pred.squeeze() == targets).sum().item()
            correct_top3 += sum([targets[i] in top3_pred[i] for i in range(len(targets))])
            total += targets.size(0)

    avg_loss = test_loss / len(test_loader)
    top1_accuracy = correct_top1 / total
    top3_accuracy = correct_top3 / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

    return avg_loss, top1_accuracy, top3_accuracy

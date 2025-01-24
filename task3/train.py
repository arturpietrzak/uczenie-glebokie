import torch


def train_classifier(model, train_loader, val_loader, criterion, optimizer, epochs, patience, name, device):
    print("Training model")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch.x = batch.x.float()
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y.squeeze(-1).type(torch.int64))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch.x = batch.x.float()
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(output, batch.y.squeeze(-1).type(torch.int64))
                optimizer.step()
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


def train_regressor(model, train_loader, val_loader, criterion, optimizer, epochs, patience, name, device):
    print(f"Training model ${name}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0

        for batch in train_loader:
            # batch_count += 1
            # if batch_count > 100:
            #     break

            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch).view(-1, 1)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.batch).view(-1, 1)
                loss = criterion(output, batch.y)
                optimizer.step()
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

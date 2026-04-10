from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.lstm import LSTMForecast


def train_model(train_ds, val_ds, config):
    save_path = config.get("save_path")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model = LSTMForecast(H=config["H"])
    model = model.to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    loss_fn = torch.nn.L1Loss()  # MAE

    batch_size = config.get("batch_size", 1024)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val = float("inf")
    patience = 0
    max_epochs = config.get("epochs", 50)
    patience_limit = config.get("patience", 3)

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for x, y in train_loader:
            x, y = x.to(config["device"]), y.to(config["device"])

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            batch_size_actual = x.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_examples += batch_size_actual

        val_loss = evaluate(model, val_loader, config)
        train_loss = train_loss_sum / max(train_examples, 1)

        print(f"Epoch {epoch:03d}: train_mae={train_loss:.4f}, val_mae={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience >= patience_limit:
                break

    if save_path is not None and Path(save_path).exists():
        model.load_state_dict(
            torch.load(save_path, map_location=config["device"])
        )

    return model


def evaluate(model, loader, config):
    model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config["device"]), y.to(config["device"])
            y_hat = model(x)
            loss = torch.mean(torch.abs(y_hat - y), dim=1)
            total += loss.sum().item()
            count += x.size(0)

    return total / max(count, 1)

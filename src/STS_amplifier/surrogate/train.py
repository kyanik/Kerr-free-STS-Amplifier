from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import NumpyRegressionDataset
from .model import MLP


@dataclass
class Standardizer:
    mean: list[float]
    std: list[float]

    def transform(self, x: np.ndarray) -> np.ndarray:
        m = np.asarray(self.mean)
        s = np.asarray(self.std)
        return (x - m) / (s + 1e-12)

# axis=0 collapses rows and is used to get the mean and std of the columns individually.
def fit_standardizer(X: np.ndarray) -> Standardizer:
    return Standardizer(mean=X.mean(axis=0).tolist(), std=X.std(axis=0).tolist())


def train_surrogate(
    data_npz: str,
    out_dir: str = "surrogate_artifacts",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(data_npz)
    X = data["X"]
    y = data["y"]

    x_scaler = fit_standardizer(X)
    Xn = x_scaler.transform(X)

    ds = NumpyRegressionDataset(Xn, y)
    n_train = int(0.9 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLP(in_dim=X.shape[1], hidden=hidden, out_dim=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train() #training mode enables things like dropout and batchnorm, which are only used during training
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad() #clears old gradients from the last step (otherwise you'd just accumulate the gradients from all loss.backward() calls)
            loss.backward() # backpropagation computes the gradients of the loss with respect to the model's parameters
            opt.step() #updates your model's weights and parameters based on the gradients computed during backpropagation

        model.eval() #validation mode disables dropout and batchnorm, which are only used during training
        val_losses = []
        with torch.no_grad(): #do not compute gradients during validation, saves memory and computations
            for xb, yb in val_loader:
                pred = model(xb) #validation predictions with data from the validation set
                val_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "model.pt")

        print(f"epoch {epoch:03d} | val_mse={val_loss:.4e}")

    (out / "scaler.json").write_text(json.dumps(asdict(x_scaler), indent=2))
    (out / "meta.json").write_text(json.dumps({"in_dim": X.shape[1], "out_dim": y.shape[1], "hidden": hidden}, indent=2))
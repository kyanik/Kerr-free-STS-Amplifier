import json
from pathlib import Path
import numpy as np
import torch

from .model import MLP


def load_scaler(path: str):
    d = json.loads(Path(path).read_text())
    mean = np.asarray(d["mean"])
    std = np.asarray(d["std"])
    return mean, std


def predict(model_dir: str, X: np.ndarray) -> np.ndarray:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    mean, std = load_scaler(str(model_dir / "scaler.json"))

    Xn = (X - mean) / (std + 1e-12)
    Xn = torch.tensor(Xn, dtype=torch.float32)

    model = MLP(in_dim=meta["in_dim"], hidden=meta["hidden"], out_dim=meta["out_dim"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y = model(Xn).numpy()
    return y
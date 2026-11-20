"""NaN/inf guards in the shared training loop (a fp16 overflow crashed a B4 run at epoch 4)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from forensight.evaluation import metrics  # noqa: E402
from forensight.training.common import NonFiniteModelError, fit, model_is_finite, predict  # noqa: E402


class Toy(Dataset):
    """8 features, label = feature 0 > 0. Sample `nan_at` (if any) contains NaN."""

    def __init__(self, n=64, nan_at=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 8, generator=g)
        self.y = (self.x[:, 0] > 0).float()
        if nan_at is not None:
            self.x[nan_at] = float("nan")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], f"vid{i // 4}"

    def labels(self):
        return self.y.numpy()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.body(x).squeeze(1)


def cfg(epochs=2):
    return {"train": {"epochs": epochs, "lr": 1e-2, "log_every": 1000, "resume": False}}


def test_nan_batch_is_skipped_and_batchnorm_stats_stay_finite(tmp_path):
    torch.manual_seed(0)
    model = Net()
    train = DataLoader(Toy(nan_at=5), batch_size=8, shuffle=False)
    val = DataLoader(Toy(seed=1), batch_size=8)
    out = fit(model, train, val, cfg(), tmp_path)
    assert model_is_finite(model)
    assert out["history"][-1]["skipped_batches"] == 1
    assert np.isfinite(out["history"][-1]["train_loss"])
    assert (tmp_path / "checkpoint.pt").exists()


def test_predict_raises_clear_error_for_broken_model():
    model = Net()
    with torch.no_grad():
        model.body[0].weight[0, 0] = float("nan")
    with pytest.raises(NonFiniteModelError):
        predict(model, DataLoader(Toy(), batch_size=8), torch.device("cpu"))


def test_fit_stops_without_saving_when_weights_break(tmp_path):
    model = Net()
    with torch.no_grad():
        model.body[3].bias.fill_(float("nan"))
    train = DataLoader(Toy(), batch_size=8, shuffle=False)
    with pytest.raises(NonFiniteModelError):
        fit(model, train, DataLoader(Toy(seed=1), batch_size=8), cfg(1), tmp_path)
    assert not (tmp_path / "checkpoint.pt").exists()


def test_metrics_reject_nan_scores_with_a_clear_message():
    with pytest.raises(ValueError, match="NaN/inf"):
        metrics.eer([0, 1, 0, 1], [0.1, float("nan"), 0.2, 0.9])

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from terraclass.model import LandUseCNN
from terraclass.training import evaluate, set_reproducibility, train_one_epoch


def test_training_and_evaluation_smoke() -> None:
    set_reproducibility(42)
    images = torch.randn(10, 3, 32, 32)
    labels = torch.tensor([0, 1] * 5)
    loader = DataLoader(TensorDataset(images, labels), batch_size=5, shuffle=False)
    model = LandUseCNN(2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    before = model[0].weight.detach().clone()
    train_metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    after = model[0].weight.detach()
    evaluation = evaluate(model, loader, criterion, torch.device("cpu"), ("a", "b"))
    assert train_metrics["loss"] > 0
    assert 0 <= train_metrics["accuracy"] <= 1
    assert not torch.equal(before, after)
    assert 0 <= evaluation["accuracy"] <= 1
    assert 0 <= evaluation["macro_f1"] <= 1
    assert len(evaluation["confusion_matrix"]) == 2

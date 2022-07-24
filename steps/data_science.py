import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from villard import V
from villard.io import BaseDataReader
from zmq import device

from .lib.model import Model


@V.register_custom_data_reader("DT_PYTORCH_MODEL")
class PytorchModelReader(BaseDataReader):
    def read_data(self, path: str, *args, **kwargs) -> object:
        super().read_data(path, *args, **kwargs)
        return torch.load(path)


@V.node("train_model")
def train_model(
    train_data_loader: DataLoader, max_epochs: int, n_views: int, n_classes: int
) -> Model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(n_views=n_views, n_classes=n_classes).to(device)
    ce_loss_fn = torch.nn.CrossEntropyLoss().to(device)

    adam = torch.optim.Adam(model.parameters(), lr=0.001)
    pbar = tqdm(range(max_epochs))
    for epoch in pbar:
        losses = []
        for i, batch in enumerate(train_data_loader):
            count, views = batch[0].to(device), batch[1].to(device)
            pred = model(views)
            loss = ce_loss_fn(pred, count - 1)
            loss.backward()
            adam.step()
            adam.zero_grad()
            losses.append(loss.item())
        pbar.set_description(f"epoch mean loss: {np.mean(losses)}")
    torch.save(model, "data/03_output/model.pt")
    return model


@V.node("evaluate_model")
def evaluate_model(test_data_loader: DataLoader, model: Model) -> None:
    predictions = []
    actual = []
    model = model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            count, views = batch[0], batch[1]
            pred = model(views).argmax(dim=1) + 1
            predictions.extend(pred.detach().numpy().tolist())
            actual.extend(count.detach().numpy())
    print(predictions)
    print(actual)
    acc = accuracy_score(actual, predictions)
    print(acc)
    return None

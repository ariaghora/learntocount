from torch.utils.data import DataLoader

from villard import V

from .lib.dataloader import GeneratedDataset


@V.node("load_train_data")
def load_train_data(n_views: int, batch_size: int) -> DataLoader:
    dataset = GeneratedDataset("data/02_intermediate/generated_train", n_views)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


@V.node("load_test_data")
def load_test_data(n_views: int, batch_size: int) -> DataLoader:
    dataset = GeneratedDataset("data/02_intermediate/generated_test", n_views)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

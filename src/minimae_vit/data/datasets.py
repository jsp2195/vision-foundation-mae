from torchvision.datasets import CIFAR10, STL10


class UnlabeledWrapper:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        return x, -1


def build_pretrain_dataset(cfg, transform):
    if cfg.dataset.name == "stl10" or cfg.dataset.use_stl_unlabeled:
        ds = STL10(cfg.dataset.data_root, split="unlabeled", transform=transform, download=True)
        return UnlabeledWrapper(ds)
    return UnlabeledWrapper(
        CIFAR10(cfg.dataset.data_root, train=True, transform=transform, download=True)
    )


def build_train_test_datasets(cfg, train_transform, eval_transform):
    train = CIFAR10(cfg.dataset.data_root, train=True, transform=train_transform, download=True)
    test = CIFAR10(cfg.dataset.data_root, train=False, transform=eval_transform, download=True)
    return train, test

from torchvision import transforms


def build_train_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def build_eval_transform(img_size: int):
    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

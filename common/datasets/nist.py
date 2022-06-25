from torchvision.transforms import transforms

nist_transforms = {
    "train": transforms.Compose(
        [
            # transforms.RandomResizedCrop(28),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    ),
    "val": transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
}

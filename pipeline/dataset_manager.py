import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataset_loaders(dataset_name, batch_size, validation_split=0.1):
    """
    Prepare data loaders for the specified dataset.
    """
    if dataset_name == 'dataset1':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='data/dataset1', train=True, download=True, transform=transform)
    elif dataset_name == 'dataset2':
        # Define transformations and dataset for dataset2
        pass
    # Add more datasets as needed
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Split dataset into training and validation sets
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Additional dataset handling functions can be added as needed

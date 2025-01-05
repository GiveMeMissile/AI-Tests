import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Image
from transformers import AutoFeatureExtractor
from torchvision import transforms
from os import cpu_count
from tqdm import tqdm

NUM_WORKERS = cpu_count()
TRAIN_DATA = "copycat-project/laion2b6plus_fish"
PRETRAINED_EXTRACTOR = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 16
MAGNITUDE_BINS = 31
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=MAGNITUDE_BINS),
    transforms.ToTensor()
])


# I wannacry
def transform(examples):
    examples["pixel_values"] = [TRANSFORM(image) for image in examples["image"]]
    return examples


def get_data():
    dataset = load_dataset(TRAIN_DATA, split="train")
    dataset = dataset.with_format(type="torch")
    dataset = dataset.cast_column("image", Image(mode="RGB"))
    print(dataset)
    dataset = dataset.set_transform(transform)
    print(dataset)
    train_dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=NUM_WORKERS, shuffle=True)
    replicated_train_dataloader = []
    for batch in train_dataloader:
        X = torch.randn(size=(16, 3, 64, 64))
        replicated_train_dataloader.append(X)
        print(batch)
    return train_dataloader, replicated_train_dataloader


if __name__ == "__main__":
    get_data()

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Image
from torchvision import transforms
from os import cpu_count

NUM_WORKERS = cpu_count()
TRAIN_DATA = "copycat-project/laion2b6plus_fish"
BATCH_SIZE = 16
MAGNITUDE_BINS = 31
IMAGE_DIMENSIONS = 64
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(IMAGE_DIMENSIONS, IMAGE_DIMENSIONS)),
    transforms.ToTensor()
])


def transform(examples):
    examples["pixel_values"] = [TRANSFORM(image) for image in examples["image"]]
    return examples


def get_data():
    print("Getting huggingface dataset...")
    datasets = load_dataset(TRAIN_DATA, split="train[75%:]")
    datasets = datasets.cast_column("image", Image(mode="RGB"))
    datasets = datasets.with_format(type="torch")

    replicated_data = []
    category = []

    for i in range(len(datasets)):
        X = torch.rand(size=(3, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS))
        replicated_data.append(X)
        category.append("non_fish")

    datasets.set_transform(transform)
    datasets.to_dict()
    # print(datasets["category"])

    data = {"image": ((datasets["image"])["pixel_values"]) + replicated_data,
            "category": datasets["category"] + category}
    print(data["image"])
    print("finished")
    return data


if __name__ == "__main__":
    get_data()

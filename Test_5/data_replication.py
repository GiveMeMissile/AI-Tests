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


def augment_data(example):
    augmented_image = torch.randn(size=(3, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS))
    example["image"] = [example["image"], augmented_image]
    example["category"] = [example["category"], "non_fish"]
    return example


def get_data():
    print("Getting huggingface dataset...")
    datasets = load_dataset(TRAIN_DATA, split="train[10%:]")
    datasets = datasets.cast_column("image", Image(mode="RGB"))
    datasets = datasets.with_format(type="torch")

    '''
    for i in range(len(datasets)):
        X = torch.randn(size=(3, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS))
        replicated_data.append(X)
        category.append("non_fish")
    

    data = {"image": datasets["image"] + replicated_data,
            "category": datasets["category"] + category}

    print("DATA CREATED")

    replicated_dataset = datasets.from_dict(data)

    print("Transforming data...")
    replicated_dataset.data(transform)
    '''

    datasets = datasets.map(augment_data, batched=False)
    datasets.set_transform(transform)

    train_dataloader = DataLoader(dataset=datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    print("Train data has been fully collected.")
    return train_dataloader


if __name__ == "__main__":
    get_data()

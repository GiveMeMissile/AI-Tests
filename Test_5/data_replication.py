import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Image
from torchvision import transforms
from os import cpu_count
from random import shuffle

NUM_WORKERS = cpu_count()
TRAIN_DATA = "copycat-project/laion2b6plus_fish"
BATCH_SIZE = 16
MAGNITUDE_BINS = 31
IMAGE_DIMENSIONS = 64
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(IMAGE_DIMENSIONS, IMAGE_DIMENSIONS)),
    transforms.ToTensor()
])


class DataPreparer:
    def __init__(self, data):
        self.dataset = data

    def shuffle_data(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        merged_data = list(zip(dataset["image"], dataset["category"]))
        shuffle(merged_data)
        dataset["image"], dataset["category"] = zip(*merged_data)
        self.dataset["image"] = dataset["image"]
        self.dataset["category"] = dataset["category"]
        return dataset

    def batch_data(self, dataset):
        pass

    def transform(self, examples):
        examples["pixel_values"] = [TRANSFORM(image) for image in examples["image"]]
        return examples

    def synthesize_data(self):
        print("\nSynthesizing data...")
        replicated_data = []
        replicated_category = []

        for i in range(len(self.dataset)):
            X = torch.rand(size=(3, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS))
            replicated_data.append(X)
            replicated_category.append("non_fish")

        self.dataset.set_transform(self.transform)
        self.dataset.to_dict()

        pixel_values = []

        for values in self.dataset:
            pixel_values.append(values["pixel_values"])

        category = []

        for categories in self.dataset:
            category.append(categories["category"])

        data = {"image": pixel_values + replicated_data,
                "category": category + replicated_category}

        self.dataset = data

        print("Data successfully synthesized")
        return data


def get_data():
    print("Getting huggingface dataset...")
    datasets = load_dataset(TRAIN_DATA, split="train[75%:]")
    datasets = datasets.cast_column("image", Image(mode="RGB"))
    datasets = datasets.with_format(type="torch")
    print("Got dataset")

    return datasets


def main():
    data = get_data()
    data_handler = DataPreparer(data=data)
    dataset = data_handler.synthesize_data()


if __name__ == "__main__":
    main()

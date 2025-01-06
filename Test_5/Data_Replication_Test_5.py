import torch
from torch import nn
from datasets import load_dataset, Image
from torchvision import transforms
from os import cpu_count
from random import shuffle

NUM_WORKERS = cpu_count()
TRAIN_DATA = "copycat-project/laion2b6plus_fish"
BATCH_SIZE = 32
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

    def batch_data(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        dataset = self.tensorize_category(dataset)
        batch = 1
        images = []
        categories = []
        X_list = []
        y_list = []
        for i, (image, category) in enumerate(zip(dataset["image"], dataset["category"])):
            images.append(image)
            categories.append(category)
            if i+1 == BATCH_SIZE*batch or len(dataset["image"]) == i+1:
                batch += 1
                X = torch.stack(images, dim=0)
                y = torch.stack(categories, dim=0)
                images.clear()
                categories.clear()
                X_list.append(X)
                y_list.append(y)
        dataset = {"X": X_list, "y": y_list}
        self.dataset = dataset
        return dataset

    def tensorize_category(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        categories = []
        for category in dataset["category"]:
            if category == "fish":
                y = torch.tensor([1])
            else:
                y = torch.tensor([0])
            categories.append(y)

        dataset["category"] = categories
        self.dataset = dataset
        return dataset

    def transform(self, examples):
        examples["pixel_values"] = [TRANSFORM(image) for image in examples["image"]]
        return examples

    def synthesize_data(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        print("\nSynthesizing data...")
        replicated_data = []
        replicated_category = []

        for i in range(len(dataset)):
            X = torch.rand(size=(3, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS))
            replicated_data.append(X)
            replicated_category.append("non_fish")

        dataset.set_transform(self.transform)
        dataset.to_dict()

        pixel_values = []

        for values in dataset:
            pixel_values.append(values["pixel_values"])

        category = []

        for categories in dataset:
            category.append(categories["category"])

        data = {"image": pixel_values + replicated_data,
                "category": category + replicated_category}

        self.dataset = data

        print("Data successfully synthesized")
        return data

    def prepare_data(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        dataset = self.synthesize_data(dataset)
        dataset = self.shuffle_data(dataset)
        dataset = self.batch_data(dataset)
        return dataset


def get_data():
    # This function gets the required dataset from hugging face and returns it.
    print("Getting huggingface dataset...")
    datasets = load_dataset(TRAIN_DATA, split="train")
    datasets = datasets.cast_column("image", Image(mode="RGB"))
    datasets = datasets.with_format(type="torch")
    print("Got dataset")

    return datasets


def main():
    data = get_data()
    train_data_handler = DataPreparer(data=data)
    train_dataset = train_data_handler.prepare_data()


if __name__ == "__main__":
    main()

from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name="samsum"):
        self.dataset_name = dataset_name

    def load_data(self, split="train"):
        dataset = load_dataset(self.dataset_name)[split]
        return dataset
